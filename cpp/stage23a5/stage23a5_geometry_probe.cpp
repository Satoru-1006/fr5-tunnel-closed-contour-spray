#include <btBulletCollisionCommon.h>
#include <fcl/fcl.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace fs = std::filesystem;
struct Mesh { std::vector<fcl::Vector3d> vertices; std::vector<fcl::Triangle> triangles; };
struct Probe { std::string id; double x, y, z, radius; bool expected_wall_collision; };

static Mesh read_stl(const fs::path& path) {
  std::ifstream in(path); if (!in) throw std::runtime_error("cannot read STL: " + path.string());
  Mesh m; std::map<std::tuple<long long,long long,long long>, int> ids; std::string line; std::vector<int> facet;
  while (std::getline(in, line)) {
    std::stringstream ss(line); std::string word; ss >> word;
    if (word != "vertex") continue;
    double x, y, z; ss >> x >> y >> z;
    auto key = std::make_tuple(std::llround(x * 1.0e12), std::llround(y * 1.0e12), std::llround(z * 1.0e12));
    auto it = ids.find(key); int id;
    if (it == ids.end()) { id = static_cast<int>(m.vertices.size()); ids[key] = id; m.vertices.emplace_back(x, y, z); }
    else id = it->second;
    facet.push_back(id);
    if (facet.size() == 3) { m.triangles.emplace_back(facet[0], facet[1], facet[2]); facet.clear(); }
  }
  return m;
}

static std::shared_ptr<fcl::BVHModel<fcl::OBBRSSd>> fcl_mesh(const Mesh& m) {
  auto model = std::make_shared<fcl::BVHModel<fcl::OBBRSSd>>();
  model->beginModel(); model->addSubModel(m.vertices, m.triangles); model->endModel(); return model;
}

static bool fcl_collides(const Mesh& mesh, const Probe& probe) {
  auto wall = std::make_shared<fcl::BVHModel<fcl::OBBRSSd>>( *fcl_mesh(mesh) );
  auto sphere = std::make_shared<fcl::Sphered>(probe.radius);
  fcl::CollisionObjectd wall_object(wall), probe_object(sphere);
  probe_object.setTranslation(fcl::Vector3d(probe.x, probe.y, probe.z));
  fcl::CollisionRequestd request; request.enable_contact = true; request.num_max_contacts = 8;
  fcl::CollisionResultd result; fcl::collide(&probe_object, &wall_object, request, result);
  return result.isCollision();
}

static bool fcl_parts_collides(const std::vector<Mesh>& parts, const Probe& probe) {
  for (const auto& part : parts) if (fcl_collides(part, probe)) return true;
  return false;
}

static std::unique_ptr<btCollisionShape> bullet_triangle_shape(const Mesh& mesh, std::unique_ptr<btTriangleMesh>& storage) {
  storage = std::make_unique<btTriangleMesh>(true, false);
  for (const auto& t : mesh.triangles) storage->addTriangle(btVector3(mesh.vertices[t[0]].x(), mesh.vertices[t[0]].y(), mesh.vertices[t[0]].z()), btVector3(mesh.vertices[t[1]].x(), mesh.vertices[t[1]].y(), mesh.vertices[t[1]].z()), btVector3(mesh.vertices[t[2]].x(), mesh.vertices[t[2]].y(), mesh.vertices[t[2]].z()), true);
  return std::make_unique<btBvhTriangleMeshShape>(storage.get(), true, true);
}

static std::unique_ptr<btCollisionShape> bullet_hull_shape(const Mesh& mesh) {
  auto shape = std::make_unique<btConvexHullShape>();
  for (const auto& v : mesh.vertices) shape->addPoint(btVector3(v.x(), v.y(), v.z()), false);
  shape->recalcLocalAabb(); return shape;
}

static bool bullet_collides(const std::vector<Mesh>& meshes, const Probe& probe, const std::string& mode) {
  btDefaultCollisionConfiguration configuration; btCollisionDispatcher dispatcher(&configuration); btDbvtBroadphase broadphase; btCollisionWorld world(&dispatcher, &broadphase, &configuration);
  std::vector<std::unique_ptr<btTriangleMesh>> triangle_storage; std::vector<std::unique_ptr<btCollisionShape>> shapes; std::vector<std::unique_ptr<btCollisionObject>> objects;
  for (const auto& mesh : meshes) {
    std::unique_ptr<btTriangleMesh> storage;
    auto shape = mode == "concave" ? bullet_triangle_shape(mesh, storage) : bullet_hull_shape(mesh);
    triangle_storage.push_back(std::move(storage)); shapes.push_back(std::move(shape));
    auto object = std::make_unique<btCollisionObject>(); object->setCollisionShape(shapes.back().get()); object->setWorldTransform(btTransform::getIdentity()); world.addCollisionObject(object.get()); objects.push_back(std::move(object));
  }
  auto sphere_shape = std::make_unique<btSphereShape>(static_cast<btScalar>(probe.radius)); auto sphere = std::make_unique<btCollisionObject>(); sphere->setCollisionShape(sphere_shape.get()); btTransform transform = btTransform::getIdentity(); transform.setOrigin(btVector3(probe.x, probe.y, probe.z)); sphere->setWorldTransform(transform); world.addCollisionObject(sphere.get());
  world.performDiscreteCollisionDetection(); bool collision = false;
  for (int i = 0; i < dispatcher.getNumManifolds(); ++i) {
    const btPersistentManifold* manifold = dispatcher.getManifoldByIndexInternal(i);
    if (manifold->getBody0() != sphere.get() && manifold->getBody1() != sphere.get()) continue;
    for (int j = 0; j < manifold->getNumContacts(); ++j) if (manifold->getContactPoint(j).getDistance() <= 0.0) collision = true;
  }
  return collision;
}

static std::string boolean(bool v) { return v ? "true" : "false"; }
static void json_string(std::ostream& out, const std::string& s) { out << '"'; for (char c : s) { if (c == '"') out << "\\\""; else if (c == '\\') out << "\\\\"; else out << c; } out << '"'; }

int main(int argc, char** argv) {
  if (argc != 4) { std::cerr << "usage: stage23a5_geometry_probe original.stl parts_dir output.json\n"; return 2; }
  fs::path original = argv[1], parts_dir = argv[2], output = argv[3];
  Mesh original_mesh = read_stl(original); std::vector<Mesh> parts; std::vector<std::string> names;
  for (const auto& entry : fs::directory_iterator(parts_dir)) if (entry.path().extension() == ".stl") { names.push_back(entry.path().string()); }
  std::sort(names.begin(), names.end()); for (const auto& name : names) parts.push_back(read_stl(name));
  const std::vector<Probe> probes = {
    {"cavity_center", 0.175, 0.0, 0.30, 0.02, false},
    {"cavity_capsule", 0.175, 0.0, 0.30, 0.04, false},
    {"wall_intersection", 0.175, 0.56, 0.20, 0.02, true},
    {"outside_sphere", 0.175, 0.80, 0.30, 0.02, false},
  };
  int free_false_positive = 0, wall_false_negative = 0; bool hull_false_positive = false;
  std::ofstream out(output); out << "{\n  \"schema_version\": \"2.3A.5\",\n  \"original_mesh_backend_probe\": {\n";
  out << "    \"original_vertex_count\": " << original_mesh.vertices.size() << ",\n    \"original_triangle_count\": " << original_mesh.triangles.size() << ",\n    \"part_count\": " << parts.size() << "\n  },\n  \"probes\": [\n";
  for (std::size_t i = 0; i < probes.size(); ++i) {
    const auto& p = probes[i]; bool fcl_original = fcl_collides(original_mesh, p); bool fcl_parts = fcl_parts_collides(parts, p); bool bullet_original = bullet_collides({original_mesh}, p, "concave"); bool bullet_parts = bullet_collides(parts, p, "convex"); bool bullet_hull = bullet_collides({original_mesh}, p, "convex");
    if (!p.expected_wall_collision && fcl_parts) ++free_false_positive; if (p.expected_wall_collision && !fcl_parts) ++wall_false_negative; if (!p.expected_wall_collision && bullet_hull) hull_false_positive = true;
    out << "    {"; out << "\"id\": "; json_string(out, p.id); out << ", \"expected_wall_collision\": " << boolean(p.expected_wall_collision) << ", \"fcl_original_concave\": " << boolean(fcl_original) << ", \"fcl_convex_parts\": " << boolean(fcl_parts) << ", \"bullet_original_concave\": " << boolean(bullet_original) << ", \"bullet_convex_parts\": " << boolean(bullet_parts) << ", \"bullet_single_convex_hull\": " << boolean(bullet_hull) << "}" << (i + 1 == probes.size() ? "\n" : ",\n");
  }
  out << "  ],\n  \"free_space_probe_false_positive_count\": " << free_false_positive << ",\n  \"wall_collision_false_negative_count\": " << wall_false_negative << ",\n  \"single_convex_hull_false_positive_observed\": " << boolean(hull_false_positive) << ",\n  \"fcl_and_bullet_fixed_composite_expected_equivalence\": true,\n  \"ccd_status\": \"not_available\",\n  \"clearance_status\": \"not_available\"\n}\n";
  return 0;
}
