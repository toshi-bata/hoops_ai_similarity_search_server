// scs_points - author a HOOPS Web Viewer point-cloud SCS from a JSON group list.
//
// This is a thin, offline authoring tool. The server computes the Shape Map
// (one point per file, projected to [-1, 1] and coloured) and calls this tool to
// bake those points into a .scs file that the Web Viewer can open by URL.
//
// Structure: the points are split into CATEGORY GROUPS (one k-means cluster, or
// one tag, or "untagged"). Each group becomes ONE mesh + ONE named node, so the
// viewer can show/hide a whole category by node visibility and could recolour a
// node later. The colour is carried by the node's POINT MATERIAL (one colour per
// node), not per-point RGBA, which is what makes node-level recolouring possible.
//
// Contract with the server (do not "improve" by adding identity fields here):
//   * Point ORDER within a group, and GROUP order, are the only link back to a
//     file_id. The server emits groups and their points in the same order as its
//     /map response, and names each node; the viewer maps a picked (node, point
//     index) pair to (group offset + point index) -> /map points array.
//   * The tool therefore never receives file_id or cluster - only geometry,
//     colour, and a node name. Passing identity here would mislead a future
//     maintainer into thinking the tool resolves identity itself; it does not.
//
// Input JSON:
//   { "version": 2,
//     "groups": [
//       { "node_name": "group_0000", "r":<0-255>, "g":<0-255>, "b":<0-255>,
//         "points": [ { "x":<f>, "y":<f>, "z":<f> }, ... ] }, ... ] }
// Colours are pre-computed by the server; the tool has no idea what a colour or a
// node name means (matching the legend/ids is the server's responsibility).
//
// CLI:  scs_points --input <points.json> --output <out.scs>
//   success: exit 0, one status line on stdout
//   failure: non-zero exit, one reason line on stderr

#include "hoops_license.h"
#include "sc_store.h"
#include "sc_assemblytree.h"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class ApplicationLogger : public SC::Store::Logger {
public:
    void Message(const char* message) const override { std::fprintf(stderr, "%s\n", message); }
};

struct PointXYZ {
    double x = 0.0, y = 0.0, z = 0.0;
};

struct Group {
    std::string node_name;
    int r = 0, g = 0, b = 0;
    std::vector<PointXYZ> points;
};

// --- Minimal JSON reader -------------------------------------------------
// A small, dependency-free parser for exactly the schema above. It supports the
// subset of JSON the server emits (objects, arrays, numbers, strings, booleans,
// null) and is deliberately strict about structure but lenient about whitespace.
class JsonParser {
public:
    explicit JsonParser(const std::string& text) : _s(text), _i(0) {}

    // Parse the top-level object and return its "groups". Throws std::runtime_error
    // on any structural problem so the caller can report a single clear reason.
    std::vector<Group> parseGroups() {
        skipWs();
        expect('{');
        std::vector<Group> groups;
        bool sawGroups = false;
        for (;;) {
            skipWs();
            if (peek() == '}') { get(); break; }
            std::string key = parseString();
            skipWs();
            expect(':');
            if (key == "groups") {
                groups = parseGroupArray();
                sawGroups = true;
            } else {
                skipValue();  // ignore "version" and any future sibling fields
            }
            skipWs();
            if (peek() == ',') { get(); continue; }
        }
        if (!sawGroups) throw std::runtime_error("input JSON has no 'groups' array");
        return groups;
    }

private:
    std::vector<Group> parseGroupArray() {
        skipWs();
        expect('[');
        std::vector<Group> out;
        skipWs();
        if (peek() == ']') { get(); return out; }
        for (;;) {
            out.push_back(parseGroupObject());
            skipWs();
            char c = get();
            if (c == ']') break;
            if (c != ',') throw std::runtime_error("malformed 'groups' array");
        }
        return out;
    }

    Group parseGroupObject() {
        skipWs();
        expect('{');
        Group grp;
        for (;;) {
            skipWs();
            if (peek() == '}') { get(); break; }
            std::string key = parseString();
            skipWs();
            expect(':');
            skipWs();
            if (key == "node_name") {
                grp.node_name = parseString();
            } else if (key == "points") {
                grp.points = parsePointArray();
            } else if (key == "r") {
                grp.r = clampByte(parseNumber());
            } else if (key == "g") {
                grp.g = clampByte(parseNumber());
            } else if (key == "b") {
                grp.b = clampByte(parseNumber());
            } else {
                skipValue();
            }
            skipWs();
            if (peek() == ',') { get(); continue; }
        }
        return grp;
    }

    std::vector<PointXYZ> parsePointArray() {
        skipWs();
        expect('[');
        std::vector<PointXYZ> out;
        skipWs();
        if (peek() == ']') { get(); return out; }
        for (;;) {
            out.push_back(parsePointObject());
            skipWs();
            char c = get();
            if (c == ']') break;
            if (c != ',') throw std::runtime_error("malformed 'points' array");
        }
        return out;
    }

    PointXYZ parsePointObject() {
        skipWs();
        expect('{');
        PointXYZ p;
        for (;;) {
            skipWs();
            if (peek() == '}') { get(); break; }
            std::string key = parseString();
            skipWs();
            expect(':');
            skipWs();
            double v = parseNumber();
            if (key == "x") p.x = v;
            else if (key == "y") p.y = v;
            else if (key == "z") p.z = v;
            skipWs();
            if (peek() == ',') { get(); continue; }
        }
        return p;
    }

    static int clampByte(double v) {
        long n = static_cast<long>(v);
        if (n < 0) n = 0;
        if (n > 255) n = 255;
        return static_cast<int>(n);
    }

    // Value skipper for keys we do not consume (keeps the parser total).
    void skipValue() {
        skipWs();
        char c = peek();
        if (c == '{') { skipContainer('{', '}'); }
        else if (c == '[') { skipContainer('[', ']'); }
        else if (c == '"') { parseString(); }
        else {
            // number, true, false, null
            while (_i < _s.size()) {
                char d = _s[_i];
                if (d == ',' || d == '}' || d == ']' || isWs(d)) break;
                _i++;
            }
        }
    }

    void skipContainer(char open, char close) {
        expect(open);
        int depth = 1;
        while (_i < _s.size() && depth > 0) {
            char c = get();
            if (c == '"') { rewind(); parseString(); continue; }
            if (c == open) depth++;
            else if (c == close) depth--;
        }
        if (depth != 0) throw std::runtime_error("unbalanced JSON container");
    }

    std::string parseString() {
        skipWs();
        expect('"');
        std::string out;
        while (_i < _s.size()) {
            char c = _s[_i++];
            if (c == '"') return out;
            if (c == '\\') {
                if (_i >= _s.size()) break;
                char e = _s[_i++];
                switch (e) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'n': out.push_back('\n'); break;
                    case 't': out.push_back('\t'); break;
                    case 'r': out.push_back('\r'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'u': _i += 4; out.push_back('?'); break;  // keys/values here are ASCII
                    default: out.push_back(e); break;
                }
            } else {
                out.push_back(c);
            }
        }
        throw std::runtime_error("unterminated JSON string");
    }

    double parseNumber() {
        skipWs();
        size_t start = _i;
        while (_i < _s.size()) {
            char c = _s[_i];
            if ((c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E') {
                _i++;
            } else {
                break;
            }
        }
        if (_i == start) throw std::runtime_error("expected a number");
        return std::stod(_s.substr(start, _i - start));
    }

    static bool isWs(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }
    void skipWs() { while (_i < _s.size() && isWs(_s[_i])) _i++; }
    char peek() { return _i < _s.size() ? _s[_i] : '\0'; }
    char get() { return _i < _s.size() ? _s[_i++] : '\0'; }
    void rewind() { if (_i > 0) _i--; }
    void expect(char c) {
        skipWs();
        if (get() != c) throw std::runtime_error(std::string("expected '") + c + "' in JSON");
    }

    const std::string& _s;
    size_t _i;
};

std::string readFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open input file: " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string argValue(int argc, char** argv, const std::string& flag) {
    for (int i = 1; i < argc - 1; ++i) {
        if (flag == argv[i]) return argv[i + 1];
    }
    return std::string();
}

}  // namespace

int main(int argc, char** argv) {
    const std::string input = argValue(argc, argv, "--input");
    const std::string output = argValue(argc, argv, "--output");
    if (input.empty() || output.empty()) {
        std::fprintf(stderr, "usage: scs_points --input <points.json> --output <out.scs>\n");
        return 2;
    }

    ApplicationLogger logger;
    std::filesystem::path workdir = std::filesystem::path(output).string() + ".work";

    try {
        std::vector<Group> groups;
        {
            const std::string text = readFile(input);
            JsonParser parser(text);
            groups = parser.parseGroups();
        }
        size_t total_points = 0;
        for (const auto& g : groups) total_points += g.points.size();
        if (total_points == 0) throw std::runtime_error("input has zero points");

        SC::Store::Database::SetLicense(HOOPS_LICENSE);
        SC::Store::Cache cache = SC::Store::Database::Open(logger);

        // A private, unique working store; the final single-file .scs is emitted by
        // GenerateSCSFile and the working directory is removed afterwards.
        std::error_code ec;
        std::filesystem::remove_all(workdir, ec);
        std::filesystem::create_directories(workdir, ec);
        const std::string modelPath = (workdir / "model").string();

        SC::Store::Model model = cache.Open(modelPath.c_str());
        SC::Store::InclusionKey inclusion_key = model.Include(model);

        // Author a named node per group so the viewer can address a whole category
        // (show/hide, recolour). The node NAME is the stable key back to the group;
        // the client must not rely on node creation order.
        SC::Store::AssemblyTree assembly_tree(logger);
        uint32_t root_id = 0;
        assembly_tree.CreateAssemblyTreeRoot(root_id);
        assembly_tree.SetNodeName(root_id, "shape_map");

        for (const auto& grp : groups) {
            if (grp.points.empty()) continue;

            // Coordinates are passed through unchanged: the server already
            // normalises the Shape Map to the [-1, 1] cube and the Web Viewer sizes
            // point splats relative to the bounding diagonal, so no unit conversion
            // is needed here.
            std::vector<SC::Store::Point> points;
            points.reserve(grp.points.size());
            for (const auto& p : grp.points) {
                points.emplace_back(static_cast<float>(p.x), static_cast<float>(p.y),
                                    static_cast<float>(p.z));
            }

            // One mesh of bare points ('P' elements, no per-point RGBA). Colour is
            // carried by the instance's POINT MATERIAL below, so the whole node is a
            // single colour that a viewer can later change with setNodesColors.
            SC::Store::Mesh mesh;
            mesh.points = points.data();
            mesh.point_count = static_cast<uint32_t>(points.size());

            SC::Store::MeshElement point_element;
            point_element.indices.reserve(points.size());
            for (uint32_t i = 0; i < points.size(); ++i) {
                point_element.indices.push_back(i);
            }
            mesh.point_elements.push_back(std::move(point_element));
            mesh.flags = static_cast<SC::Store::Mesh::Bits>(
                SC::Store::Mesh::CounterClockwiseWinding);

            SC::Store::MeshKey mesh_key = model.Insert(mesh);

            SC::Store::Color colour(static_cast<float>(grp.r) / 255.0f,
                                    static_cast<float>(grp.g) / 255.0f,
                                    static_cast<float>(grp.b) / 255.0f, 1.0f);
            SC::Store::MaterialKey material_key = model.Insert(colour);

            // point_material paints every point in this node the group's colour.
            SC::Store::InstanceKey instance_key = model.Instance(
                mesh_key, SC::Store::MatrixKey(), SC::Store::MaterialKey::Unspecified(),
                SC::Store::MaterialKey::Unspecified(), material_key);

            // Named child node -> body instance -> this mesh instance. Name both the
            // child and the body-instance node with the group name so a pick lands
            // on a named node regardless of which level the viewer reports.
            uint32_t child_node = 0;
            assembly_tree.CreateChild(root_id, child_node);
            assembly_tree.SetNodeName(child_node, grp.node_name.c_str());
            uint32_t body_instance_node = 0;
            assembly_tree.CreateAndAddBodyInstance(child_node, body_instance_node);
            assembly_tree.SetNodeName(body_instance_node, grp.node_name.c_str());
            assembly_tree.SetBodyInstanceMeshInstanceKey(
                body_instance_node, SC::Store::InstanceInc(inclusion_key, instance_key));
        }

        assembly_tree.SerializeToModel(model);
        model.PrepareStream();

        const char* written = model.GenerateSCSFile(output.c_str());
        if (written == nullptr) throw std::runtime_error("GenerateSCSFile returned no path");

        cache.Close(model);
        std::filesystem::remove_all(workdir, ec);

        std::printf("scs_points: wrote %zu points in %zu groups to %s\n", total_points,
                    groups.size(), output.c_str());
        return 0;
    } catch (const std::exception& e) {
        std::error_code ec;
        std::filesystem::remove_all(workdir, ec);
        std::fprintf(stderr, "scs_points error: %s\n", e.what());
        return 1;
    }
}
