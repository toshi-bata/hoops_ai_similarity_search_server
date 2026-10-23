// CAD file extensions the server accepts, mirrored from
// core.CAD_ALLOWED_EXTENSIONS (the server is the authority; this copy only
// pre-filters a chosen folder so non-CAD files -- images, text, CAM output --
// are never hashed or uploaded). If the two ever drift, the server still
// rejects anything unexpected per-file in upload-batch, so the worst case is a
// wasted upload, not a wrong result. Keep in sync when the server list changes.
//
// Lower-case, leading dot. Callers lower-case the candidate suffix before
// testing (SolidWorks often writes upper-case, e.g. part.SLDPRT).
export const CAD_EXTENSIONS = new Set([
  '.3ds', '.3mf', '.3dxml', '.sat', '.sab', '.dgn',
  '.dwg', '.dxf', '.dwf', '.dwfx', '.model', '.dlv',
  '.exp', '.session', '.catpart', '.catproduct', '.catshape',
  '.catdrawing', '.cgr', '.dae', '.prt', '.neu',
  '.asm', '.xas', '.xpr', '.fbx', '.gltf', '.glb',
  '.arc', '.unv', '.mf1', '.pkg', '.ifc', '.ifczip',
  '.igs', '.iges', '.ipt', '.iam', '.jt', '.kmz',
  '.nwd', '.prc', '.x_t', '.xmt', '.x_b', '.xmt_txt',
  '.rvt', '.3dm', '.stp', '.step', '.stpz', '.stl',
  '.par', '.pwd', '.psm', '.sldprt', '.sldasm', '.sldfpp',
  '.u3d', '.vda', '.wrl', '.vrml', '.obj', '.xv3',
  '.xv0', '.hsf',
]);

// True when a file name ends in a CAD extension this server accepts. Handles
// compound extensions like ".x_t"/".xmt_txt" by testing suffixes rather than
// just the text after the last dot.
export function isCadFile(name) {
  const lower = (name || '').toLowerCase();
  for (const ext of CAD_EXTENSIONS) {
    if (lower.endsWith(ext)) return true;
  }
  return false;
}
