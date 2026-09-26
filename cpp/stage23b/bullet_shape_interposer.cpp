// Project-local runtime overlay for Stage 2.3B.
//
// The installed Jazzy MoveIt2 Bullet implementation already has a maintained
// USE_SHAPE_TYPE mesh path.  The current allocator selects CONVEX_HULL for
// meshes, which expands non-convex link occupancy.  This interposer changes
// only that runtime representation when explicitly enabled and delegates the
// actual construction, ownership, and Bullet semantics to the installed
// MoveIt2 implementation.

#include <dlfcn.h>

#include <cstdlib>
#include <stdexcept>
#include <string>

#include <BulletCollision/CollisionShapes/btCollisionShape.h>
#include <geometric_shapes/shapes.h>
#include <moveit/collision_detection_bullet/bullet_integration/basic_types.hpp>
#include <moveit/collision_detection_bullet/bullet_integration/bullet_utils.hpp>

namespace collision_detection_bullet
{
namespace
{
using MeshFactory = btCollisionShape* (*)(const shapes::Mesh*, const CollisionObjectType&,
                                          CollisionObjectWrapper*);

constexpr const char* kMeshFactorySymbol =
    "_ZN26collision_detection_bullet20createShapePrimitiveEPKN6shapes4MeshERKNS_19CollisionObjectTypeEPNS_22CollisionObjectWrapperE";

MeshFactory original_mesh_factory()
{
  static MeshFactory factory = [] {
    void* symbol = dlsym(RTLD_NEXT, kMeshFactorySymbol);
    if (symbol == nullptr)
      throw std::runtime_error("Stage 2.3B could not resolve installed MoveIt2 Bullet mesh factory");
    return reinterpret_cast<MeshFactory>(symbol);
  }();
  return factory;
}

bool enabled()
{
  const char* value = std::getenv("FR5_BULLET_SHAPE_MODE");
  return value != nullptr && std::string(value) == "use_shape_type";
}

bool is_disputed_link(const CollisionObjectWrapper* cow)
{
  if (cow == nullptr)
    return false;
  const std::string& name = cow->getName();
  return name == "forearm_link" || name == "wrist2_link" || name == "wrist3_link";
}
}  // namespace

btCollisionShape* createShapePrimitive(const shapes::Mesh* mesh, const CollisionObjectType& requested,
                                       CollisionObjectWrapper* cow)
{
  if (!enabled() || !is_disputed_link(cow))
    return original_mesh_factory()(mesh, requested, cow);

  // Delegate to the installed, version-matched implementation.  For meshes,
  // USE_SHAPE_TYPE is its triangle-compound representation; no source mesh,
  // transform, padding, ACM, or contact threshold is changed here.
  return original_mesh_factory()(mesh, CollisionObjectType::USE_SHAPE_TYPE, cow);
}
}  // namespace collision_detection_bullet
