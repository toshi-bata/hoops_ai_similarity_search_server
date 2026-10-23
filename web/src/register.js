// Bulk registration workspace: turn a folder (or a set of files, or the single
// file open in the viewer) into an asynchronous embedding job, then follow that
// job to completion.
//
// Everything funnels into one server pipeline:
//   1. hash each file locally (hash.js) and ask which the server already has
//      (POST /files/exists), so only missing files are uploaded;
//   2. upload the missing ones (POST /files/upload-batch, in chunks);
//   3. queue one job with every file_id (POST .../jobs/add) and poll it
//      (GET .../jobs/{id}) until it reaches a terminal state;
//   4. show the plain-text report and offer to resubmit the retryable failures
//      as a second job -- the documented two-pass loop.
//
// The individual "Register open file" action (a file opened via "Open local
// file..." that is not yet indexed) is the same pipeline with the upload
// already done: it holds a file_id, so it skips straight to step 3.

import * as api from './api.js';
import { ApiError } from './api.js';
import { sha256Hex } from './hash.js';
import { isCadFile } from './cad-extensions.js';

// Files per upload-batch request. Keeps each multipart request bounded on a
// large folder while still amortising the per-request overhead.
const UPLOAD_CHUNK = 40;
// Job status poll interval. The server parses the SDK's own progress bar, so a
// second is frequent enough to feel live without hammering the endpoint.
const POLL_MS = 1000;
// Terminal job statuses (mirror jobstore.TERMINAL_STATUSES).
const TERMINAL = new Set(['done', 'failed', 'canceled']);

// The overall progress bar spans the whole pipeline, not just the embedding
// job, so a bulk run never looks frozen during the local prep that precedes
// it. These are the cumulative end points (percent of the whole run) of the
// prep stages; the embedding job fills the remainder (STAGE_QUEUE_END..100).
const STAGE_HASH_END = 5;    // local hashing of the selection
const STAGE_UPLOAD_END = 30; // uploading the files the server is missing
const STAGE_QUEUE_END = 35;  // queuing the job (job then runs 35..100)

// Create the registration controller. `getIndexName` returns the index the
// workspace acts on; `onJobComplete` is called after a job settles so the host
// can refresh the gallery and index stats to show the newly added parts.
export function createRegister({ getIndexName, onJobComplete }) {
  const el = (id) => document.getElementById(id);

  const modal = el('register-modal');
  const indexLabel = el('register-index');
  const selectionInfo = el('register-selection');
  const submitButton = el('register-submit');
  const phaseLine = el('register-phase');
  const progressBox = el('register-progress');
  const barFill = el('register-bar-fill');
  const progressText = el('register-progress-text');
  const cancelButton = el('register-cancel');
  const resultBox = el('register-result');
  const summaryLine = el('register-summary');
  const resubmitButton = el('register-resubmit');
  const reportPre = el('register-report');
  const jobsList = el('register-jobs-list');
  const folderInput = el('register-folder-input');
  const filesInput = el('register-files-input');
  const dropZone = el('register-drop');

  // Files chosen for a bulk run (CAD only), keyed by name for a simple count.
  let selectedFiles = [];
  // The job currently being polled, and its poll timer, so opening a new run or
  // closing the modal cancels the old poll instead of leaking timers.
  let pollTimer = null;
  // file_ids whose failures are worth resubmitting, captured when a job ends.
  let retryableIds = [];

  // The share of the overall bar reserved for the embedding job. Bulk runs do
  // the hash/upload/queue prep first, so their job fills STAGE_QUEUE_END..100;
  // the single-file and resubmit paths have no prep, so their job fills 0..100.
  let embedBase = 0;
  let embedSpan = 100;

  function clamp01(x) {
    return Math.max(0, Math.min(1, x));
  }

  // Drive the one overall progress bar. `indeterminate` sweeps an animated
  // stripe across the whole track for steps that cannot report a percentage
  // (the exists check, queuing, and a job that has not started embedding yet),
  // so those steps read as "working" rather than "stuck". The Cancel button
  // only makes sense once a job exists, so it is hidden during prep.
  function setOverall(pct, text, { indeterminate = false, showCancel = false } = {}) {
    progressBox.hidden = false;
    barFill.style.width = `${Math.max(0, Math.min(100, pct))}%`;
    progressBox.classList.toggle('is-indeterminate', indeterminate);
    cancelButton.hidden = !showCancel;
    if (text != null) progressText.textContent = text;
  }

  function stopPolling() {
    if (pollTimer) {
      clearTimeout(pollTimer);
      pollTimer = null;
    }
  }

  function setPhase(text, isError = false) {
    phaseLine.textContent = text || '';
    phaseLine.classList.toggle('is-error', isError);
  }

  function resetRunUi() {
    stopPolling();
    progressBox.hidden = true;
    progressBox.classList.remove('is-indeterminate');
    resultBox.hidden = true;
    resubmitButton.hidden = true;
    reportPre.textContent = '';
    summaryLine.textContent = '';
    barFill.style.width = '0%';
    progressText.textContent = '';
    cancelButton.hidden = true;
    retryableIds = [];
    setPhase('');
  }

  // -- File selection (bulk) ------------------------------------------------

  function setSelection(files) {
    selectedFiles = files;
    if (files.length === 0) {
      selectionInfo.textContent = '';
      submitButton.disabled = true;
      return;
    }
    selectionInfo.textContent = `${files.length} CAD file(s) selected.`;
    submitButton.disabled = false;
  }

  // Keep only CAD files (a folder pick returns everything under it). Accepts a
  // FileList or an array.
  function filterCad(fileList) {
    return Array.from(fileList).filter((f) => isCadFile(f.name));
  }

  function onFolderChosen(event) {
    resetRunUi();
    setSelection(filterCad(event.target.files));
  }

  // -- The run: hash -> exists -> upload -> queue -> poll -------------------

  async function uploadMissing(files) {
    // Hash every file; group by whether the server already holds it. If the
    // browser exposes no crypto.subtle (non-secure origin), skip the check and
    // upload everything -- correctness over the bandwidth optimisation.
    setOverall(0, `Hashing 0 / ${files.length}\u2026`, { indeterminate: files.length === 0 });
    let hashed = 0;
    const hashes = await Promise.all(files.map(async (f) => {
      const h = await sha256Hex(f);
      hashed += 1;
      // Repaint periodically (and on the last one) so a large folder shows the
      // bar creeping up instead of jumping only when hashing is fully done.
      if (hashed === files.length || hashed % 20 === 0) {
        setOverall(
          (hashed / files.length) * STAGE_HASH_END,
          `Hashing ${hashed} / ${files.length}\u2026`,
        );
      }
      return h;
    }));
    const canHash = hashes.every((h) => h !== null);

    let known = [];
    let toUpload = files;
    let invalidCount = 0;
    if (canHash) {
      setOverall(STAGE_HASH_END, `Checking 0 / ${hashes.length}\u2026`, {
        indeterminate: hashes.length === 0,
      });
      // api.existsFiles chunks the ids itself; the callback reports real
      // counts so a folder of 10k files does not look frozen.
      const result = await api.existsFiles(hashes, (done, total) => {
        setOverall(STAGE_HASH_END, `Checking ${done} / ${total}\u2026`);
      });
      const knownSet = new Set(result.known || []);
      known = (result.known || []).slice();
      // `invalid` ids are a distinct signal from `unknown` (server-side hash
      // format check) -- surface the count rather than silently re-uploading
      // them alongside genuinely-missing files.
      invalidCount = (result.invalid || []).length;
      toUpload = files.filter((_, i) => !knownSet.has(hashes[i]));
    }

    const uploadedIds = [];
    const uploadErrors = [];
    const uploadSpan = STAGE_UPLOAD_END - STAGE_HASH_END;
    setOverall(STAGE_HASH_END, `Uploading 0 / ${toUpload.length}\u2026`, {
      indeterminate: toUpload.length === 0,
    });
    for (let i = 0; i < toUpload.length; i += UPLOAD_CHUNK) {
      const chunk = toUpload.slice(i, i + UPLOAD_CHUNK);
      const res = await api.uploadBatch(chunk);
      for (const f of res.files || []) uploadedIds.push(f.file_id);
      for (const e of res.errors || []) uploadErrors.push(e);
      const uploaded = Math.min(i + chunk.length, toUpload.length);
      setOverall(
        STAGE_HASH_END + (uploaded / toUpload.length) * uploadSpan,
        `Uploading ${uploaded} / ${toUpload.length}\u2026`,
      );
    }

    return { fileIds: [...known, ...uploadedIds], uploadErrors, invalidCount };
  }

  // Returns { options } on success, or { error } if an entered value fails
  // the server's constraints (both fields are `ge: 1`).
  function readOptions() {
    const workersRaw = el('register-workers').value.trim();
    const timeRaw = el('register-timelimit').value.trim();
    const options = {};
    if (workersRaw) {
      const workers = Number(workersRaw);
      if (!Number.isFinite(workers) || workers < 1) return { error: 'Workers must be a number \u2265 1.' };
      options.workers = workers;
    }
    if (timeRaw) {
      const timeLimit = Number(timeRaw);
      if (!Number.isFinite(timeLimit) || timeLimit < 1) return { error: 'Time limit must be a number \u2265 1.' };
      options.timeLimit = timeLimit;
    }
    return { options };
  }

  // Queue a job for the given file_ids and follow it. Shared by the bulk run,
  // the individual file action, and resubmit -- each caller resets the run UI
  // itself before invoking this, since they need control over what phase text
  // shows first (e.g. "Hashing…" or "Registering…").
  async function queueAndFollow(name, fileIds, options, { embedFrom = 0, embedTo = 100 } = {}) {
    if (fileIds.length === 0) {
      setPhase('No files to register.', true);
      return;
    }
    // Reserve this stretch of the overall bar for the embedding job so the
    // follow-up polling continues the same bar rather than restarting it.
    embedBase = embedFrom;
    embedSpan = Math.max(0, embedTo - embedFrom);
    setOverall(embedBase, `Queuing a job for ${fileIds.length} file(s)\u2026`, {
      indeterminate: true,
    });
    let accepted;
    try {
      accepted = await api.createIndexJob(name, fileIds, options);
    } catch (err) {
      setPhase(describeError(err), true);
      return;
    }
    setPhase('');
    followJob(name, accepted.job_id);
  }

  async function runBulk() {
    const name = getIndexName();
    if (!name || selectedFiles.length === 0) return;
    const { options, error } = readOptions();
    if (error) {
      resetRunUi();
      setPhase(error, true);
      return;
    }
    submitButton.disabled = true;
    try {
      resetRunUi();
      const { fileIds, uploadErrors, invalidCount } = await uploadMissing(selectedFiles);
      const warnings = [];
      if (uploadErrors.length) warnings.push(`${uploadErrors.length} file(s) could not be uploaded`);
      if (invalidCount) warnings.push(`${invalidCount} file(s) had an invalid id and were re-uploaded`);
      if (warnings.length) setPhase(`${warnings.join('; ')}; continuing with the rest.`);
      await queueAndFollow(name, fileIds, options, { embedFrom: STAGE_QUEUE_END, embedTo: 100 });
    } catch (err) {
      setPhase(describeError(err), true);
    } finally {
      submitButton.disabled = selectedFiles.length === 0;
    }
  }

  // -- Polling --------------------------------------------------------------

  function renderProgress(job) {
    const p = job.progress || {};
    const total = p.total || 0;
    const done = p.done || 0;
    // Until the SDK reports a total the job is still starting up (queued or
    // spinning up its workers): hold the bar at the job's start point and sweep
    // the indeterminate stripe so this stretch does not look frozen.
    const started = total > 0;
    const overall = embedBase + (started ? clamp01(done / total) * embedSpan : 0);
    const indeterminate = !started && !TERMINAL.has(job.status);
    const bits = [`${job.status}`, `phase: ${p.phase || '-'}`, `${done} / ${total}`];
    if (p.errors) bits.push(`${p.errors} error(s)`);
    if (p.heavy) bits.push(`${p.heavy} heavy`);
    setOverall(overall, bits.join('  \u00b7  '), { indeterminate, showCancel: true });
    // Cancellation only bites before embedding starts; once running it may not
    // stop, so reflect that a request is already in flight.
    cancelButton.disabled = job.cancel_requested;
    cancelButton.textContent = job.cancel_requested ? 'Cancel requested' : 'Cancel';
  }

  async function followJob(name, jobId) {
    stopPolling();
    cancelButton.dataset.jobId = jobId;
    const tick = async () => {
      let job;
      try {
        job = await api.getIndexJob(name, jobId);
      } catch (err) {
        setPhase(describeError(err), true);
        return;
      }
      renderProgress(job);
      if (TERMINAL.has(job.status)) {
        await onJobSettled(name, job);
        return;
      }
      pollTimer = setTimeout(tick, POLL_MS);
    };
    tick();
  }

  async function onJobSettled(name, job) {
    stopPolling();
    progressBox.hidden = true;
    resultBox.hidden = false;

    const s = job.summary || {};
    const summaryBits = [
      `Status: ${job.status}`,
      `added: ${s.added ?? 0}`,
      `failed: ${s.failed ?? 0}`,
    ];
    if (s.retryable) summaryBits.push(`retryable: ${s.retryable}`);
    if (s.heavy_flagged) summaryBits.push(`heavy: ${s.heavy_flagged}`);
    if (job.detail) summaryBits.push(job.detail);
    summaryLine.textContent = summaryBits.join('  \u00b7  ');

    retryableIds = (job.errors || [])
      .filter((e) => e.retryable && e.file_id)
      .map((e) => e.file_id);
    resubmitButton.hidden = retryableIds.length === 0;
    if (retryableIds.length) {
      resubmitButton.textContent = `Resubmit ${retryableIds.length} retryable failure(s)`;
    }

    // The report exists only once the job wrote one (report_url set). Fetch it
    // as text; a 404 just means there is nothing to show.
    if (job.report_url) {
      try {
        reportPre.textContent = await api.getIndexJobReport(name, job.job_id);
      } catch {
        reportPre.textContent = '';
      }
    }

    await refreshJobs(name);
    if (typeof onJobComplete === 'function') {
      try {
        await onJobComplete();
      } catch {
        // A refresh failure in the host must not break the workspace.
      }
    }
  }

  async function resubmit() {
    const name = getIndexName();
    if (!name || retryableIds.length === 0) return;
    // The retry pass: fewer workers, a longer per-file budget (the caller can
    // still override in the option fields before resubmitting).
    const { options, error } = readOptions();
    if (error) {
      setPhase(error, true);
      return;
    }
    // Capture before resetRunUi() clears retryableIds.
    const ids = retryableIds;
    resetRunUi();
    await queueAndFollow(name, ids, options);
  }

  // -- Jobs list ------------------------------------------------------------

  async function refreshJobs(name) {
    try {
      const { jobs } = await api.listIndexJobs(name, { limit: 20 });
      renderJobs(name, jobs || []);
    } catch {
      jobsList.textContent = '';
    }
  }

  function renderJobs(name, jobs) {
    jobsList.innerHTML = '';
    if (jobs.length === 0) {
      jobsList.textContent = 'No registration jobs yet.';
      return;
    }
    for (const job of jobs) {
      const row = document.createElement('div');
      row.className = 'register-job-row';

      const p = job.progress || {};
      const s = job.summary || {};
      const label = document.createElement('span');
      label.className = 'register-job-label';
      const counts = TERMINAL.has(job.status)
        ? `added ${s.added ?? 0}, failed ${s.failed ?? 0}`
        : `${p.done || 0} / ${p.total || 0}`;
      label.textContent = `${job.status}  \u00b7  ${counts}`;
      row.appendChild(label);

      const when = job.created_at ? job.created_at.replace('T', ' ').replace('Z', '') : '';
      const time = document.createElement('span');
      time.className = 'register-job-time';
      time.textContent = when;
      row.appendChild(time);

      // Follow an active job, or reopen a finished one's report.
      const follow = document.createElement('button');
      follow.type = 'button';
      follow.textContent = TERMINAL.has(job.status) ? 'View' : 'Follow';
      follow.addEventListener('click', () => {
        if (TERMINAL.has(job.status)) {
          onJobSettled(name, job).catch(() => {});
        } else {
          followJob(name, job.job_id);
        }
      });
      row.appendChild(follow);

      jobsList.appendChild(row);
    }
  }

  // -- Modal open/close -----------------------------------------------------

  function open() {
    const name = getIndexName();
    if (!name) return;
    resetRunUi();
    setSelection([]);
    el('register-workers').value = '';
    el('register-timelimit').value = '';
    indexLabel.textContent = `Index: ${name}`;
    modal.hidden = false;
    refreshJobs(name);
  }

  function close() {
    stopPolling();
    modal.hidden = true;
  }

  // Individual mode: a file already uploaded (opened in the viewer) that is not
  // yet in the index. It holds a file_id, so skip selection/upload and queue.
  function registerCurrentFile(fileId, filename) {
    const name = getIndexName();
    if (!name || !fileId) return;
    resetRunUi();
    setSelection([]);
    el('register-workers').value = '';
    el('register-timelimit').value = '';
    indexLabel.textContent = `Index: ${name}`;
    modal.hidden = false;
    setPhase(`Registering "${filename || fileId}"\u2026`);
    queueAndFollow(name, [fileId], {});
  }

  // -- Wiring ---------------------------------------------------------------

  el('register-close').addEventListener('click', close);
  el('register-pick-folder').addEventListener('click', () => folderInput.click());
  el('register-pick-files').addEventListener('click', () => filesInput.click());
  folderInput.addEventListener('change', onFolderChosen);
  filesInput.addEventListener('change', onFolderChosen);
  submitButton.addEventListener('click', () => runBulk().catch(() => {}));
  resubmitButton.addEventListener('click', () => resubmit().catch(() => {}));
  cancelButton.addEventListener('click', async () => {
    const name = getIndexName();
    if (!name) return;
    // job_id lives on the polled job; re-list is unnecessary since cancel needs
    // the current job. Track it via a data attribute set in followJob.
    const jobId = cancelButton.dataset.jobId;
    if (!jobId) return;
    try {
      await api.cancelIndexJob(name, jobId);
    } catch (err) {
      setPhase(describeError(err), true);
    }
  });

  // Drag-and-drop onto the drop zone (files only; folders via the picker).
  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('is-drag');
  });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('is-drag'));
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('is-drag');
    resetRunUi();
    setSelection(filterCad(e.dataTransfer.files));
  });

  // Close when clicking the backdrop (outside the dialog).
  modal.addEventListener('click', (e) => {
    if (e.target === modal) close();
  });

  return { open, close, registerCurrentFile };
}

// Turn any thrown error into a one-line message (server detail for ApiError).
// Deliberately not routed through ui.js's withLoading()/showError(): this
// modal has its own inline phase/progress line rather than the page-level
// loading overlay, so errors surface there instead.
function describeError(err) {
  if (err instanceof ApiError) return `${err.status}: ${err.detail}`;
  return err && err.message ? err.message : String(err);
}
