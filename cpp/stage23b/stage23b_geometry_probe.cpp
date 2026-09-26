// Stage 2.3B direct geometry regression for the selected triangle-compound
// representation.  Reuse the frozen Stage 2.3A.5 mesh/probe definitions and
// add the representation that the upstream USE_SHAPE_TYPE mesh factory emits.

#define main stage23a5_geometry_probe_main
#include "../stage23a5/stage23a5_geometry_probe.cpp"
#undef main
#include <BulletCollision/CollisionShapes/btTriangleShape.h>

static bool bullet_compound_collides(const Mesh& mesh, const Probe& probe)
{
  btDefaultCollisionConfiguration configuration;
  btCollisionDispatcher dispatcher(&configuration);
  btDbvtBroadphase broadphase;
  btCollisionWorld world(&dispatcher, &broadphase, &configuration);

  auto compound = std::make_unique<btCompoundShape>(true, static_cast<int>(mesh.triangles.size()));
  std::vector<std::unique_ptr<btTriangleShape>> triangles;
  triangles.reserve(mesh.triangles.size());
  for (const auto& triangle : mesh.triangles)
  {
    auto child = std::make_unique<btTriangleShape>(
        btVector3(mesh.vertices[triangle[0]].x(), mesh.vertices[triangle[0]].y(), mesh.vertices[triangle[0]].z()),
        btVector3(mesh.vertices[triangle[1]].x(), mesh.vertices[triangle[1]].y(), mesh.vertices[triangle[1]].z()),
        btVector3(mesh.vertices[triangle[2]].x(), mesh.vertices[triangle[2]].y(), mesh.vertices[triangle[2]].z()));
    child->setMargin(btScalar(0));
    compound->addChildShape(btTransform::getIdentity(), child.get());
    triangles.push_back(std::move(child));
  }
  compound->recalculateLocalAabb();

  btCollisionObject mesh_object;
  mesh_object.setCollisionShape(compound.get());
  mesh_object.setWorldTransform(btTransform::getIdentity());
  world.addCollisionObject(&mesh_object);

  btSphereShape sphere_shape(static_cast<btScalar>(probe.radius));
  btCollisionObject sphere;
  sphere.setCollisionShape(&sphere_shape);
  btTransform transform = btTransform::getIdentity();
  transform.setOrigin(btVector3(probe.x, probe.y, probe.z));
  sphere.setWorldTransform(transform);
  world.addCollisionObject(&sphere);
  world.performDiscreteCollisionDetection();

  for (int i = 0; i < dispatcher.getNumManifolds(); ++i)
  {
    const btPersistentManifold* manifold = dispatcher.getManifoldByIndexInternal(i);
    if (manifold->getBody0() != &sphere && manifold->getBody1() != &sphere)
      continue;
    for (int j = 0; j < manifold->getNumContacts(); ++j)
      if (manifold->getContactPoint(j).getDistance() <= btScalar(0))
        return true;
  }
  return false;
}

int main(int argc, char** argv)
{
  if (argc != 3)
  {
    std::cerr << "usage: stage23b_geometry_probe original.stl output.json\n";
    return 2;
  }
  const Mesh mesh = read_stl(argv[1]);
  const std::vector<Probe> probes = {
      { "cavity_center", 0.175, 0.0, 0.30, 0.02, false },
      { "cavity_capsule", 0.175, 0.0, 0.30, 0.04, false },
      { "wall_intersection", 0.175, 0.56, 0.20, 0.02, true },
      { "outside_sphere", 0.175, 0.80, 0.30, 0.02, false },
  };
  bool passed = true;
  std::ofstream out(argv[2]);
  out << "{\n  \"schema_version\": \"2.3B.direct-triangle-compound-probe\",\n"
         "  \"representation\": \"btCompoundShape_of_btTriangleShape_margin_0\",\n"
         "  \"child_shape_count\": "
      << mesh.triangles.size() << ",\n  \"cases\": [\n";
  for (std::size_t i = 0; i < probes.size(); ++i)
  {
    const auto& probe = probes[i];
    const bool independent_expected = probe.expected_wall_collision;
    const bool fcl = fcl_collides(mesh, probe);
    const bool old_hull = bullet_collides({ mesh }, probe, "convex");
    const bool new_compound = bullet_compound_collides(mesh, probe);
    const bool case_passed = fcl == independent_expected && new_compound == independent_expected;
    passed = passed && case_passed;
    out << "    {\"id\":";
    json_string(out, probe.id);
    out << ",\"independent_expected_collision\":" << boolean(independent_expected)
        << ",\"fcl_collision\":" << boolean(fcl) << ",\"old_whole_mesh_convex_hull_collision\":"
        << boolean(old_hull) << ",\"new_triangle_compound_collision\":" << boolean(new_compound)
        << ",\"passed\":" << boolean(case_passed) << "}" << (i + 1 == probes.size() ? "\n" : ",\n");
  }
  out << "  ],\n  \"passed\": " << boolean(passed)
      << ",\n  \"ccd\": \"not_available\",\n  \"clearance\": \"not_available\"\n}\n";
  return passed ? 0 : 1;
}
