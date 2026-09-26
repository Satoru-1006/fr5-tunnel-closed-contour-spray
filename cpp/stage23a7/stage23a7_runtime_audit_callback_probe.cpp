// Callback-level Bullet audit executable.  The original probe is included so
// its frozen-input readers and MoveIt helpers are reused without changing the
// production repository inputs.
#define protected public
#define main stage23a7_original_probe_main
#include "stage23a7_runtime_audit_probe.cpp"
#undef main
#ifdef checkSelfCollision
#undef checkSelfCollision
#endif

#include <openssl/sha.h>
#include <BulletCollision/CollisionDispatch/btManifoldResult.h>
#include <cstring>
#include <cstdint>
#include <limits>
#include <unordered_map>

using collision_detection_bullet::CollisionObjectWrapper;

struct SourceMeta {
  std::string path;
  std::string sha256;
  std::size_t vertex_count = 0;
  std::size_t triangle_count = 0;
};

static std::string sha256_file(const fs::path& p)
{
  if (!fs::exists(p))
    return "not_available_source_file_missing";
  SHA256_CTX c;
  SHA256_Init(&c);
  std::ifstream f(p, std::ios::binary);
  std::array<char, 1 << 16> b{};
  while (f) {
    f.read(b.data(), b.size());
    if (f.gcount() > 0)
      SHA256_Update(&c, b.data(), static_cast<std::size_t>(f.gcount()));
  }
  unsigned char digest[SHA256_DIGEST_LENGTH];
  SHA256_Final(digest, &c);
  std::ostringstream s;
  s << std::hex << std::setfill('0');
  for (unsigned char x : digest)
    s << std::setw(2) << static_cast<unsigned int>(x);
  return s.str();
}

static SourceMeta source_meta(const fs::path& p)
{
  static std::unordered_map<std::string, SourceMeta> cache;
  const std::string key = p.empty() ? "" : p.generic_string();
  const auto cached = cache.find(key);
  if (cached != cache.end())
    return cached->second;
  SourceMeta m;
  m.path = p.empty() ? "not_available" : p.generic_string();
  if (p.empty() || !fs::exists(p)) {
    m.sha256 = "not_available_source_file_missing";
    cache.emplace(key, m);
    return m;
  }
  m.sha256 = sha256_file(p);
  std::ifstream f(p, std::ios::binary);
  std::uint32_t binary_triangles = 0;
  f.seekg(80);
  f.read(reinterpret_cast<char*>(&binary_triangles), sizeof(binary_triangles));
  const auto file_size = fs::file_size(p);
  if (f && binary_triangles > 0 && 84ULL + 50ULL * binary_triangles <= file_size) {
    m.triangle_count = binary_triangles;
    std::set<std::array<std::uint32_t, 3>> unique_vertices;
    f.seekg(84);
    for (std::uint32_t t = 0; t < binary_triangles; ++t) {
      std::array<char, 50> record{};
      f.read(record.data(), record.size());
      if (!f) break;
      for (int v = 0; v < 3; ++v) {
        std::array<std::uint32_t, 3> bits{};
        std::memcpy(&bits[0], record.data() + 12 + v * 12 + 0, 4);
        std::memcpy(&bits[1], record.data() + 12 + v * 12 + 4, 4);
        std::memcpy(&bits[2], record.data() + 12 + v * 12 + 8, 4);
        unique_vertices.insert(bits);
      }
    }
    m.vertex_count = unique_vertices.size();
  } else {
    f.clear(); f.seekg(0);
    std::string line;
    while (std::getline(f, line)) {
      std::stringstream ss(line); std::string word; ss >> word;
      if (word == "vertex") ++m.vertex_count;
    }
    m.triangle_count = m.vertex_count / 3;
  }
  cache.emplace(key, m);
  return m;
}

static fs::path source_path(const fs::path& repo_root, const fs::path& parts_dir,
                            const CollisionObjectWrapper& cow, int index)
{
  if (cow.getName() == "horseshoe_collision_compound") {
    std::vector<fs::path> ps;
    for (const auto& e : fs::directory_iterator(parts_dir))
      if (e.path().extension() == ".stl")
        ps.push_back(e.path());
    std::sort(ps.begin(), ps.end());
    if (index >= 0 && index < static_cast<int>(ps.size()))
      return ps[static_cast<std::size_t>(index)];
    return {};
  }
  static const std::set<std::string> allowed = {"base_link", "shoulder_link", "upperarm_link",
                                                 "forearm_link", "wrist1_link", "wrist2_link", "wrist3_link"};
  if (allowed.count(cow.getName()))
    return repo_root / "external/frcobot_ros2/fairino_description/meshes/fairino5_v6" /
           (cow.getName() + ".STL");
  return {};
}

static btTransform eigen_bt(const Eigen::Isometry3d& t)
{
  btMatrix3x3 r(static_cast<btScalar>(t(0, 0)), static_cast<btScalar>(t(0, 1)), static_cast<btScalar>(t(0, 2)),
                static_cast<btScalar>(t(1, 0)), static_cast<btScalar>(t(1, 1)), static_cast<btScalar>(t(1, 2)),
                static_cast<btScalar>(t(2, 0)), static_cast<btScalar>(t(2, 1)), static_cast<btScalar>(t(2, 2)));
  return btTransform(r, btVector3(static_cast<btScalar>(t(0, 3)), static_cast<btScalar>(t(1, 3)),
                                  static_cast<btScalar>(t(2, 3))));
}

static std::string scalar_hex(btScalar x)
{
  std::ostringstream s;
  s << "0x" << std::hex << std::setfill('0');
  if constexpr (sizeof(btScalar) == sizeof(std::uint32_t)) {
    std::uint32_t u = 0;
    std::memcpy(&u, &x, sizeof(u));
    s << std::setw(8) << u;
  } else {
    std::uint64_t u = 0;
    std::memcpy(&u, &x, sizeof(u));
    s << std::setw(16) << u;
  }
  return s.str();
}

static std::string scalar_decimal(btScalar x)
{
  std::ostringstream s;
  s << std::setprecision(std::numeric_limits<btScalar>::max_digits10) << x;
  return s.str();
}

static void bt_vec_json(std::ostream& o, const btVector3& v)
{
  o << '[' << scalar_decimal(v.x()) << ',' << scalar_decimal(v.y()) << ',' << scalar_decimal(v.z()) << ']';
}

static void bt_vec_hex_json(std::ostream& o, const btVector3& v)
{
  o << '['; js(o, scalar_hex(v.x())); o << ','; js(o, scalar_hex(v.y())); o << ','; js(o, scalar_hex(v.z())); o << ']';
}

static void bt_transform_json(std::ostream& o, const btTransform& t)
{
  btf(o, t);
}

static void optional_eigen_transform(std::ostream& o, const CollisionObjectWrapper& cow, int index)
{
  if (index >= 0 && index < static_cast<int>(cow.shape_poses_.size()))
    tf(o, cow.shape_poses_[static_cast<std::size_t>(index)]);
  else
    o << "null";
}

static SourceMeta source_for(const fs::path& repo_root, const fs::path& parts_dir,
                             const CollisionObjectWrapper& cow, int index)
{
  return source_meta(source_path(repo_root, parts_dir, cow, index));
}

static const btCollisionShape* child_shape(const btCollisionShape* parent, int index, btTransform& child_local)
{
  child_local.setIdentity();
  if (parent && parent->getShapeType() == COMPOUND_SHAPE_PROXYTYPE) {
    const auto* c = static_cast<const btCompoundShape*>(parent);
    if (index >= 0 && index < c->getNumChildShapes()) {
      child_local = c->getChildTransform(index);
      return c->getChildShape(index);
    }
  }
  return parent;
}

struct CallbackContext {
  std::ofstream* out = nullptr;
  std::string run_id, invocation_id, node_id, waypoint_id, candidate_id;
  std::uint64_t callback_index = 0;
  std::uint64_t contact_added_index = 0;
  fs::path repo_root, parts_dir;
};

static CallbackContext* g_callback_context = nullptr;

static std::string runtime_algorithm_type(const CollisionObjectWrapper& a, const CollisionObjectWrapper& b)
{
  btDefaultCollisionConfiguration config;
  btCollisionDispatcher dispatcher(&config);
  btCollisionObjectWrapper wa(nullptr, a.getCollisionShape(), &a, a.getWorldTransform(), -1, -1);
  btCollisionObjectWrapper wb(nullptr, b.getCollisionShape(), &b, b.getWorldTransform(), -1, -1);
  btCollisionAlgorithm* algorithm = dispatcher.findAlgorithm(&wa, &wb, nullptr, BT_CLOSEST_POINT_ALGORITHMS);
  if (!algorithm) return "not_available_no_algorithm";
  const std::string result = demangle(typeid(*algorithm).name());
  dispatcher.freeCollisionAlgorithm(algorithm);
  return result;
}

static void write_callback_point(CallbackContext& ctx, const CollisionObjectWrapper& a,
                                 const CollisionObjectWrapper& b, const btManifoldPoint& cp, int contact_index,
                                 const std::string& algorithm_type)
{
  const auto* pa = a.getCollisionShape();
  const auto* pb = b.getCollisionShape();
  btTransform child_local_a, child_local_b;
  const int child_index_a = pa && pa->getShapeType() == COMPOUND_SHAPE_PROXYTYPE ? cp.m_index0 : -1;
  const int child_index_b = pb && pb->getShapeType() == COMPOUND_SHAPE_PROXYTYPE ? cp.m_index1 : -1;
  const auto* ca = child_shape(pa, child_index_a, child_local_a);
  const auto* cb = child_shape(pb, child_index_b, child_local_b);
  const SourceMeta sa = source_for(ctx.repo_root, ctx.parts_dir, a, child_index_a < 0 ? 0 : child_index_a);
  const SourceMeta sb = source_for(ctx.repo_root, ctx.parts_dir, b, child_index_b < 0 ? 0 : child_index_b);
  auto& o = *ctx.out;
  o << "{\"status\":\"callback_observed\",\"run_id\":"; js(o, ctx.run_id);
  o << ",\"invocation_id\":"; js(o, ctx.invocation_id);
  o << ",\"node_id\":"; js(o, ctx.node_id);
  o << ",\"waypoint_id\":"; js(o, ctx.waypoint_id);
  o << ",\"candidate_id\":"; js(o, ctx.candidate_id);
  o << ",\"callback_index\":" << ctx.callback_index << ",\"contact_index\":" << contact_index;
  o << ",\"body_name_0\":"; js(o, a.getName());
  o << ",\"body_name_1\":"; js(o, b.getName());
  o << ",\"body_type_0\":"; js(o, btype(a.getTypeID()));
  o << ",\"body_type_1\":"; js(o, btype(b.getTypeID()));
  o << ",\"parent_shape_type_0\":" << pa->getShapeType() << ",\"parent_shape_type_1\":" << pb->getShapeType();
  o << ",\"child_index_0\":" << child_index_a << ",\"child_index_1\":" << child_index_b;
  o << ",\"child_shape_type_0\":" << (ca ? ca->getShapeType() : -1) << ",\"child_shape_type_1\":" << (cb ? cb->getShapeType() : -1);
  o << ",\"parent_margin_0\":" << scalar_decimal(pa->getMargin()) << ",\"parent_margin_0_hex\":"; js(o, scalar_hex(pa->getMargin()));
  o << ",\"parent_margin_1\":" << scalar_decimal(pb->getMargin()) << ",\"parent_margin_1_hex\":"; js(o, scalar_hex(pb->getMargin()));
  o << ",\"child_margin_0\":" << scalar_decimal(ca ? ca->getMargin() : btScalar(0));
  o << ",\"child_margin_1\":" << scalar_decimal(cb ? cb->getMargin() : btScalar(0));
  o << ",\"parent_world_pose_0\":"; bt_transform_json(o, a.getWorldTransform());
  o << ",\"parent_world_pose_1\":"; bt_transform_json(o, b.getWorldTransform());
  o << ",\"child_local_pose_0\":"; bt_transform_json(o, child_local_a);
  o << ",\"child_local_pose_1\":"; bt_transform_json(o, child_local_b);
  o << ",\"child_world_pose_0\":"; bt_transform_json(o, a.getWorldTransform() * child_local_a);
  o << ",\"child_world_pose_1\":"; bt_transform_json(o, b.getWorldTransform() * child_local_b);
  o << ",\"source_mesh_0\":"; js(o, sa.path); o << ",\"source_mesh_sha256_0\":"; js(o, sa.sha256);
  o << ",\"source_vertex_count_0\":" << sa.vertex_count << ",\"source_triangle_count_0\":" << sa.triangle_count;
  o << ",\"source_mesh_1\":"; js(o, sb.path); o << ",\"source_mesh_sha256_1\":"; js(o, sb.sha256);
  o << ",\"source_vertex_count_1\":" << sb.vertex_count << ",\"source_triangle_count_1\":" << sb.triangle_count;
  o << ",\"m_distance1\":" << scalar_decimal(cp.m_distance1) << ",\"m_distance1_hex\":"; js(o, scalar_hex(cp.m_distance1));
  o << ",\"m_positionWorldOnA\":"; bt_vec_json(o, cp.m_positionWorldOnA); o << ",\"m_positionWorldOnA_hex\":"; bt_vec_hex_json(o, cp.m_positionWorldOnA);
  o << ",\"m_positionWorldOnB\":"; bt_vec_json(o, cp.m_positionWorldOnB); o << ",\"m_positionWorldOnB_hex\":"; bt_vec_hex_json(o, cp.m_positionWorldOnB);
  o << ",\"m_normalWorldOnB\":"; bt_vec_json(o, cp.m_normalWorldOnB); o << ",\"m_normalWorldOnB_hex\":"; bt_vec_hex_json(o, cp.m_normalWorldOnB);
  o << ",\"m_partId0\":" << cp.m_partId0 << ",\"m_partId1\":" << cp.m_partId1;
  o << ",\"m_index0\":" << cp.m_index0 << ",\"m_index1\":" << cp.m_index1;
  o << ",\"runtime_algorithm_type\":"; js(o, algorithm_type);
  o << ",\"applied_impulse\":" << scalar_decimal(cp.m_appliedImpulse) << ",\"applied_impulse_hex\":"; js(o, scalar_hex(cp.m_appliedImpulse));
  o << ",\"life_time\":" << cp.m_lifeTime;
  o << ",\"contact_processing_threshold_0\":" << scalar_decimal(a.getContactProcessingThreshold());
  o << ",\"contact_processing_threshold_1\":" << scalar_decimal(b.getContactProcessingThreshold());
  o << ",\"collision_filter_group_0\":" << a.m_collisionFilterGroup << ",\"collision_filter_group_1\":" << b.m_collisionFilterGroup;
  o << ",\"collision_filter_mask_0\":" << a.m_collisionFilterMask << ",\"collision_filter_mask_1\":" << b.m_collisionFilterMask;
  o << "}\n";
}

static void audit_near_callback(btBroadphasePair& pair, btCollisionDispatcher& dispatcher,
                                const btDispatcherInfo& info)
{
  btCollisionDispatcher::defaultNearCallback(pair, dispatcher, info);
  if (!g_callback_context || !g_callback_context->out)
    return;
  ++g_callback_context->callback_index;
  const auto* p0 = pair.m_pProxy0 ? static_cast<const btCollisionObject*>(pair.m_pProxy0->m_clientObject) : nullptr;
  const auto* p1 = pair.m_pProxy1 ? static_cast<const btCollisionObject*>(pair.m_pProxy1->m_clientObject) : nullptr;
  for (int i = 0; i < dispatcher.getNumManifolds(); ++i) {
    const auto* m = dispatcher.getManifoldByIndexInternal(i);
    if (!m || (p0 && m->getBody0() != p0 && m->getBody1() != p0) ||
        (p1 && m->getBody0() != p1 && m->getBody1() != p1))
      continue;
    const auto* a = dynamic_cast<const CollisionObjectWrapper*>(m->getBody0());
    const auto* b = dynamic_cast<const CollisionObjectWrapper*>(m->getBody1());
    if (!a || !b)
      continue;
    const std::string algorithm_type = pair.m_algorithm ? demangle(typeid(*pair.m_algorithm).name()) : "not_available_no_algorithm";
    for (int j = 0; j < m->getNumContacts(); ++j)
      write_callback_point(*g_callback_context, *a, *b, m->getContactPoint(j), j, algorithm_type);
  }
}

static bool audit_contact_added_callback(btManifoldPoint& cp, const btCollisionObjectWrapper* obj0,
                                         int part0, int index0, const btCollisionObjectWrapper* obj1,
                                         int part1, int index1)
{
  if (!g_callback_context || !g_callback_context->out || !obj0 || !obj1)
    return true;
  const auto* a = dynamic_cast<const CollisionObjectWrapper*>(obj0->getCollisionObject());
  const auto* b = dynamic_cast<const CollisionObjectWrapper*>(obj1->getCollisionObject());
  if (!a || !b)
    return true;
  btManifoldPoint raw = cp;
  raw.m_partId0 = part0; raw.m_partId1 = part1; raw.m_index0 = index0; raw.m_index1 = index1;
  write_callback_point(*g_callback_context, *a, *b, raw,
                       static_cast<int>(g_callback_context->contact_added_index++),
                       runtime_algorithm_type(*a, *b));
  return true;
}

static void write_shape_node(std::ofstream& out, const fs::path& repo_root, const fs::path& parts_dir,
                             const CollisionObjectWrapper& cow, const btCollisionShape* parent,
                             const btCollisionShape* child, int source_index, int child_index,
                             const btTransform& parent_world, const btTransform& child_local,
                             double padding, double scale, std::ofstream* vertex_out,
                             std::uint64_t* vertex_offset)
{
  if (!parent || !child)
    return;
  btVector3 plo, phi, clo, chi;
  parent->getAabb(parent_world, plo, phi);
  const btTransform child_world = parent_world * child_local;
  child->getAabb(child_world, clo, chi);
  const SourceMeta sm = source_for(repo_root, parts_dir, cow, source_index);
  out << "{\"body_name\":"; js(out, cow.getName());
  out << ",\"body_type\":"; js(out, btype(cow.getTypeID()));
  out << ",\"source_moveit_shape_type\":";
  if (source_index >= 0 && source_index < static_cast<int>(cow.shapes_.size()))
    js(out, demangle(typeid(*cow.shapes_[static_cast<std::size_t>(source_index)]).name()));
  else js(out, "not_available");
  out << ",\"source_shape_index\":" << source_index << ",\"source_mesh\":"; js(out, sm.path);
  out << ",\"source_mesh_sha256\":"; js(out, sm.sha256);
  out << ",\"source_vertex_count\":" << sm.vertex_count << ",\"source_triangle_count\":" << sm.triangle_count;
  out << ",\"bullet_shape_type_id\":" << parent->getShapeType() << ",\"bullet_get_name\":"; js(out, parent->getName() ? parent->getName() : "");
  out << ",\"concrete_cpp_type\":"; js(out, demangle(typeid(*parent).name()));
  out << ",\"is_compound\":" << (parent->getShapeType() == COMPOUND_SHAPE_PROXYTYPE ? "true" : "false");
  out << ",\"child_count\":" << (parent->getShapeType() == COMPOUND_SHAPE_PROXYTYPE ? static_cast<const btCompoundShape*>(parent)->getNumChildShapes() : 0);
  out << ",\"parent_margin\":" << scalar_decimal(parent->getMargin()) << ",\"parent_margin_hex\":"; js(out, scalar_hex(parent->getMargin()));
  out << ",\"parent_local_scaling\":"; v3(out, parent->getLocalScaling());
  out << ",\"parent_local_pose\":"; optional_eigen_transform(out, cow, source_index);
  out << ",\"parent_world_pose\":"; bt_transform_json(out, parent_world);
  out << ",\"parent_aabb_min\":"; bt_vec_json(out, plo); out << ",\"parent_aabb_max\":"; bt_vec_json(out, phi);
  out << ",\"child_index\":" << child_index << ",\"child_shape_type_id\":" << child->getShapeType() << ",\"child_get_name\":"; js(out, child->getName() ? child->getName() : "");
  out << ",\"child_concrete_cpp_type\":"; js(out, demangle(typeid(*child).name()));
  out << ",\"child_margin\":" << scalar_decimal(child->getMargin()) << ",\"child_margin_hex\":"; js(out, scalar_hex(child->getMargin()));
  out << ",\"child_local_scaling\":"; v3(out, child->getLocalScaling());
  out << ",\"child_local_pose\":"; bt_transform_json(out, child_local);
  out << ",\"child_world_pose\":"; bt_transform_json(out, child_world);
  out << ",\"child_aabb_min\":"; bt_vec_json(out, clo); out << ",\"child_aabb_max\":"; bt_vec_json(out, chi);
  out << ",\"moveit_link_padding\":" << std::setprecision(17) << padding << ",\"moveit_link_scale\":" << scale;
  out << ",\"collision_filter_group\":" << cow.m_collisionFilterGroup << ",\"collision_filter_mask\":" << cow.m_collisionFilterMask;
  out << ",\"contact_processing_threshold\":" << scalar_decimal(cow.getContactProcessingThreshold());
  std::uint64_t offset = vertex_offset ? *vertex_offset : 0;
  int count = 0;
  if (vertex_out && vertex_offset && child->getShapeType() == CONVEX_HULL_SHAPE_PROXYTYPE) {
    const auto* hull = static_cast<const btConvexHullShape*>(child);
    offset = *vertex_offset;
    count = hull->getNumPoints();
    for (int i = 0; i < count; ++i) {
      const btVector3& p = hull->getUnscaledPoints()[i];
      const btScalar xyz[3] = {p.x(), p.y(), p.z()};
      vertex_out->write(reinterpret_cast<const char*>(xyz), sizeof(xyz));
    }
    *vertex_offset += static_cast<std::uint64_t>(count) * 3U * sizeof(btScalar);
  }
  out << ",\"convex_point_count\":" << count << ",\"vertex_file\":\"runtime_shape_vertices.bin\""
      << ",\"vertex_file_offset\":" << offset
      << ",\"vertex_scalar_bytes\":" << sizeof(btScalar)
      << ",\"vertex_encoding\":\"little_endian_btScalar_xyz\"}\n";
}

static void dump_complete_shapes(planning_scene::PlanningScene& scene, const fs::path& repo_root,
                                 const fs::path& parts_dir, const fs::path& outdir,
                                 const std::shared_ptr<moveit::core::RobotModel const>& model)
{
  auto* env = dynamic_cast<collision_detection::CollisionEnvBullet*>(scene.getCollisionEnvNonConst().get());
  if (!env) return;
  std::ofstream tree(outdir / "bullet_runtime_shape_tree.jsonl");
  std::ofstream manifest(outdir / "bullet_runtime_shape_manifest.jsonl");
  std::ofstream aabb(outdir / "bullet_runtime_aabbs.csv");
  std::ofstream vertex_out(outdir / "runtime_shape_vertices.bin", std::ios::binary);
  std::uint64_t vertex_offset = 0;
  aabb << "body_name,body_type,shape_type,margin,threshold,world_transform,aabb_min,aabb_max,filter_group,filter_mask\n";
  for (const auto& kv : env->manager_->getCollisionObjects()) {
    const auto& cow = *kv.second;
    const auto* root = cow.getCollisionShape();
    const double pad = cow.getTypeID() == collision_detection::BodyTypes::ROBOT_LINK ? scene.getCollisionEnv()->getLinkPadding(cow.getName()) : 0.0;
    const double scale = cow.getTypeID() == collision_detection::BodyTypes::ROBOT_LINK ? scene.getCollisionEnv()->getLinkScale(cow.getName()) : 1.0;
    btVector3 lo, hi; cow.getAABB(lo, hi);
    aabb << cow.getName() << ',' << btype(cow.getTypeID()) << ',' << root->getShapeType() << ',' << scalar_decimal(root->getMargin()) << ',' << scalar_decimal(cow.getContactProcessingThreshold()) << ',';
    btf(aabb, cow.getWorldTransform()); aabb << ','; bt_vec_json(aabb, lo); aabb << ','; bt_vec_json(aabb, hi); aabb << ',' << cow.m_collisionFilterGroup << ',' << cow.m_collisionFilterMask << '\n';
    write_shape_node(tree, repo_root, parts_dir, cow, root, root, 0, -1, cow.getWorldTransform(), btTransform::getIdentity(), pad, scale, &vertex_out, &vertex_offset);
    if (root->getShapeType() == COMPOUND_SHAPE_PROXYTYPE) {
      const auto* c = static_cast<const btCompoundShape*>(root);
      for (int i = 0; i < c->getNumChildShapes(); ++i)
        write_shape_node(tree, repo_root, parts_dir, cow, root, c->getChildShape(i), i, i, cow.getWorldTransform(), c->getChildTransform(i), pad, scale, &vertex_out, &vertex_offset);
    }
    const SourceMeta sm = source_for(repo_root, parts_dir, cow, 0);
    manifest << "{\"body_name\":"; js(manifest, cow.getName()); manifest << ",\"body_type\":"; js(manifest, btype(cow.getTypeID()));
    manifest << ",\"source_mesh\":"; js(manifest, sm.path); manifest << ",\"source_mesh_sha256\":"; js(manifest, sm.sha256);
    manifest << ",\"source_vertex_count\":" << sm.vertex_count << ",\"source_triangle_count\":" << sm.triangle_count;
    manifest << ",\"shape_type_id\":" << root->getShapeType() << ",\"shape_cpp_type\":"; js(manifest, demangle(typeid(*root).name()));
    manifest << ",\"margin\":" << scalar_decimal(root->getMargin()) << ",\"local_scaling\":"; v3(manifest, root->getLocalScaling());
    manifest << ",\"world_pose\":"; bt_transform_json(manifest, cow.getWorldTransform());
    manifest << ",\"moveit_link_padding\":" << pad << ",\"moveit_link_scale\":" << scale;
    manifest << ",\"filter_group\":" << cow.m_collisionFilterGroup << ",\"filter_mask\":" << cow.m_collisionFilterMask << ",\"contact_processing_threshold\":" << scalar_decimal(cow.getContactProcessingThreshold()) << "}\n";
  }
}

static void set_context(CallbackContext& ctx, std::ofstream& out, const std::string& run,
                        const std::string& inv, const Candidate& c, const fs::path& repo_root,
                        const fs::path& parts_dir)
{
  ctx.out = &out; ctx.run_id = run; ctx.invocation_id = inv; ctx.node_id = c.id;
  ctx.candidate_id = c.id; ctx.waypoint_id = std::to_string(c.waypoint); ctx.callback_index = 0; ctx.contact_added_index = 0;
  ctx.repo_root = repo_root; ctx.parts_dir = parts_dir;
}

static void run_acm_case(std::ofstream& cases, CallbackContext& ctx, const std::string& backend,
                         const std::string& case_id, const Candidate& c, const RobotState& state,
                         const std::string& group, planning_scene::PlanningScene& scene,
                         const collision_detection::AllowedCollisionMatrix& acm,
                         std::ofstream& raw, const fs::path& repo_root, const fs::path& parts_dir)
{
  const std::string inv = backend + "_acm_" + case_id + "_" + c.id;
  set_context(ctx, raw, "acm", inv, c, repo_root, parts_dir); g_callback_context = &ctx;
  dump_acm_case(cases, backend, case_id, c, state, group, scene, acm);
  g_callback_context = nullptr;
}

static void run_acm_case_world(std::ofstream& cases, CallbackContext& ctx, const std::string& backend,
                               const std::string& case_id, const Candidate& c, const RobotState& state,
                               const std::string& group, planning_scene::PlanningScene& scene,
                               const collision_detection::AllowedCollisionMatrix& acm,
                               std::ofstream& raw, const fs::path& repo_root, const fs::path& parts_dir)
{
  const std::string inv = backend + "_acm_world_" + case_id + "_" + c.id;
  auto r = req(group, true); collision_detection::CollisionResult cr; cr.clear();
  set_context(ctx, raw, "acm", inv, c, repo_root, parts_dir); g_callback_context = &ctx;
  scene.checkCollision(r, cr, state, acm); g_callback_context = nullptr;
  auto x = collect(cr); cases << "{\"backend\":"; js(cases, backend); cases << ",\"case_id\":"; js(cases, case_id);
  cases << ",\"candidate_id\":"; js(cases, c.id); cases << ",\"waypoint_id\":" << c.waypoint << ",\"collision\":" << (x.collision ? "true" : "false") << ",\"pairs\":"; pairs(cases, x.pairs); cases << "}\n";
}

using OriginalAddSingleResult = btScalar (*)(collision_detection_bullet::BroadphaseContactResultCallback*,
                                              btManifoldPoint&, const btCollisionObjectWrapper*, int, int,
                                              const btCollisionObjectWrapper*, int, int);

static OriginalAddSingleResult resolve_original_add_single_result()
{
  static OriginalAddSingleResult fn = nullptr;
  static bool attempted = false;
  if (attempted) return fn;
  attempted = true;
  constexpr const char* symbol = "_ZN26collision_detection_bullet31BroadphaseContactResultCallback15addSingleResultER15btManifoldPointPK24btCollisionObjectWrapperiiS5_ii";
  fn = reinterpret_cast<OriginalAddSingleResult>(dlsym(RTLD_NEXT, symbol));
  if (!fn) {
    void* h = dlopen("/opt/ros/jazzy/lib/libmoveit_collision_detection_bullet.so.2.12.4", RTLD_LAZY | RTLD_LOCAL);
    if (h) fn = reinterpret_cast<OriginalAddSingleResult>(dlsym(h, symbol));
  }
  return fn;
}

namespace collision_detection_bullet
{
btScalar BroadphaseContactResultCallback::addSingleResult(
    btManifoldPoint& cp, const btCollisionObjectWrapper* obj0, int part0, int index0,
    const btCollisionObjectWrapper* obj1, int part1, int index1)
{
  if (g_callback_context && g_callback_context->out && obj0 && obj1) {
    const auto* a = dynamic_cast<const CollisionObjectWrapper*>(obj0->getCollisionObject());
    const auto* b = dynamic_cast<const CollisionObjectWrapper*>(obj1->getCollisionObject());
    if (a && b) {
      btManifoldPoint raw = cp;
      raw.m_partId0 = part0; raw.m_partId1 = part1; raw.m_index0 = index0; raw.m_index1 = index1;
      write_callback_point(*g_callback_context, *a, *b, raw,
                           static_cast<int>(g_callback_context->contact_added_index++),
                           runtime_algorithm_type(*a, *b));
    }
  }
  const auto fn = resolve_original_add_single_result();
  if (!fn) return btScalar(0);
  return fn(this, cp, obj0, part0, index0, obj1, part1, index1);
}
}  // namespace collision_detection_bullet

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("stage23a7_runtime_audit_callback_probe", rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));
  std::string candidate_csv, parts_dir_s, output_dir_s, backend, group, disputed_manifest, repo_root_s;
  int run = 1;
  bool only_disputed = false;
  node->get_parameter_or("candidate_csv", candidate_csv, std::string{});
  node->get_parameter_or("parts_dir", parts_dir_s, std::string{});
  node->get_parameter_or("output_dir", output_dir_s, std::string{});
  node->get_parameter_or("backend", backend, std::string("bullet"));
  node->get_parameter_or("run_index", run, 1);
  node->get_parameter_or("group_name", group, std::string("fairino5_v6_group"));
  node->get_parameter_or("disputed_manifest", disputed_manifest, std::string{});
  node->get_parameter_or("repo_root", repo_root_s, std::string("/mnt/c/Users/86198/Desktop/robotfucker"));
  node->get_parameter_or("only_disputed", only_disputed, false);
  if (candidate_csv.empty() || parts_dir_s.empty() || output_dir_s.empty()) return 2;
  const fs::path output_dir(output_dir_s), repo_root(repo_root_s), parts_dir(parts_dir_s);
  fs::create_directories(output_dir);
  const auto disputed = disputed_manifest.empty() ? std::set<std::string>{} : read_ids(disputed_manifest);
  const auto candidates = read_candidates(candidate_csv);
  const auto meshes = read_parts(parts_dir);
  robot_model_loader::RobotModelLoader::Options opt("robot_description"); opt.load_kinematics_solvers = false;
  robot_model_loader::RobotModelLoader loader(node, opt); auto model = loader.getModel(); if (!model) return 3;
  auto scene = std::make_unique<planning_scene::PlanningScene>(model);
  scene->processCollisionObjectMsg(world_object(meshes));
  if (backend == "bullet") scene->allocateCollisionDetector(collision_detection::CollisionDetectorAllocatorBullet::create());
  else scene->allocateCollisionDetector(collision_detection::CollisionDetectorAllocatorFCL::create());
  auto* jmg = model->getJointModelGroup(group); if (!jmg) return 4;
  auto& formal = scene->getAllowedCollisionMatrix();
  std::ofstream reqout(output_dir / "collision_request_manifest.jsonl", std::ios::app);
  std::ofstream stateout(output_dir / "robot_state_update_audit.jsonl", std::ios::app);
  std::ofstream tfout(output_dir / "runtime_link_transforms.jsonl", std::ios::app);
  std::ofstream rawout(output_dir / "bullet_runtime_manifolds.jsonl", std::ios::app);
  std::ofstream caseout(output_dir / "acm_case_results.jsonl", std::ios::app);
  std::ofstream full(output_dir / (backend + "_runtime_results_run" + std::to_string(run) + ".jsonl"));
  std::ofstream prov(output_dir / ("runtime_provenance_run" + std::to_string(run) + ".json"));
  std::ifstream maps("/proc/self/maps"); std::ofstream mapout(output_dir / ("proc_self_maps_run" + std::to_string(run) + ".txt")); mapout << maps.rdbuf();
  prov << "{\"status\":\"native_callback_runtime_observed\",\"backend\":"; js(prov, backend);
  prov << ",\"active_detector_name\":"; js(prov, scene->getCollisionDetectorName());
  prov << ",\"collision_environment_concrete_class\":"; js(prov, demangle(typeid(*scene->getCollisionEnv()).name()));
  prov << ",\"collision_detector_allocator\":"; js(prov, demangle(typeid(*scene->getCollisionEnv()).name()));
  prov << ",\"sizeof_btScalar\":" << sizeof(btScalar) << ",\"bt_use_double_precision\":" << (sizeof(btScalar) == 8 ? "true" : "false") << ",\"compiler\":\"gcc_runtime\",\"build_type\":\"O2\",\"ros_distro\":\"jazzy\",\"moveit_version\":\"2.12.4\",\"bullet_version\":\"3.24\",\"formal_acm_modified\":false,\"manifold_capture\":\"collision_detection_bullet::BroadphaseContactResultCallback::addSingleResult_interposed\"}\n";
  collision_detection::CollisionEnvBullet* env = backend == "bullet" ? dynamic_cast<collision_detection::CollisionEnvBullet*>(scene->getCollisionEnvNonConst().get()) : nullptr;
  btCollisionDispatcher* dispatcher = (env && env->manager_ && env->manager_->dispatcher_) ? env->manager_->dispatcher_.get() : nullptr;
  const btNearCallback previous = dispatcher ? dispatcher->getNearCallback() : nullptr;
  const ContactAddedCallback previous_contact_added = gContactAddedCallback;
  if (dispatcher) dispatcher->setNearCallback(&audit_near_callback);
  if (backend == "bullet") gContactAddedCallback = &audit_contact_added_callback;
  if (backend == "bullet") dump_complete_shapes(*scene, repo_root, parts_dir, output_dir, model);
  CallbackContext ctx;
  for (const auto& c : candidates) {
    if (only_disputed && !disputed.count(c.id)) continue;
    RobotState state(model); state.setVariablePositions(c.q);
    const bool link_before = state.dirtyLinkTransforms(), body_before = state.dirtyCollisionBodyTransforms();
    state.updateCollisionBodyTransforms();
    const bool link_after = state.dirtyLinkTransforms(), body_after = state.dirtyCollisionBodyTransforms();
    const std::string inv = backend + "_" + std::to_string(run) + "_" + c.id;
    auto r0 = req(group, false), r1 = req(group, true);
    write_request(reqout, inv + "_R0", backend, "checkCollision", r0);
    write_request(reqout, inv + "_R1", backend, "checkCollision", r1);
    collision_detection::CollisionResult a, b; a.clear();
    set_context(ctx, rawout, "run" + std::to_string(run), inv + "_R0", c, repo_root, parts_dir); g_callback_context = &ctx; scene->checkCollision(r0, a, state); g_callback_context = nullptr;
    b.clear(); set_context(ctx, rawout, "run" + std::to_string(run), inv + "_R1", c, repo_root, parts_dir); g_callback_context = &ctx; scene->checkCollision(r1, b, state); g_callback_context = nullptr;
    const auto x = collect(a), y = collect(b);
    full << "{\"candidate_id\":"; js(full, c.id); full << ",\"waypoint_id\":" << c.waypoint << ",\"joint_values\":[";
    for (std::size_t i = 0; i < c.q.size(); ++i) { if (i) full << ','; full << jn(c.q[i]); }
    full << "],\"backend\":"; js(full, backend); full << ",\"r0_collision\":" << (x.collision ? "true" : "false") << ",\"r1_collision\":" << (y.collision ? "true" : "false") << ",\"r0_pairs\":"; pairs(full, x.pairs); full << ",\"r1_pairs\":"; pairs(full, y.pairs); full << ",\"r0_contacts\":"; write_contact_json(full, x.contacts); full << ",\"r1_contacts\":"; write_contact_json(full, y.contacts); full << ",\"request_anomaly\":" << ((x.collision != y.collision) ? "true" : "false") << ",\"joint_vector_sha256\":\"computed_by_python\",\"dirty_link_before\":" << (link_before ? "true" : "false") << ",\"dirty_collision_body_before\":" << (body_before ? "true" : "false") << ",\"dirty_link_after\":" << (link_after ? "true" : "false") << ",\"dirty_collision_body_after\":" << (body_after ? "true" : "false") << ",\"collision_body_update_called\":true}\n";
    stateout << "{\"backend\":"; js(stateout, backend); stateout << ",\"run_index\":" << run << ",\"node_id\":"; js(stateout, c.id); stateout << ",\"waypoint_id\":" << c.waypoint << ",\"satisfies_bounds\":" << (state.satisfiesBounds(jmg) ? "true" : "false") << ",\"dirty_link_transforms_before\":" << (link_before ? "true" : "false") << ",\"dirty_collision_body_transforms_before\":" << (body_before ? "true" : "false") << ",\"dirty_link_transforms_after\":" << (link_after ? "true" : "false") << ",\"dirty_collision_body_transforms_after\":" << (body_after ? "true" : "false") << ",\"collision_body_update_called\":true}\n";
    for (const auto& ln : {"forearm_link", "wrist2_link", "wrist3_link"}) {
      auto* l = model->getLinkModel(ln); if (!l) continue;
      tfout << "{\"backend\":"; js(tfout, backend); tfout << ",\"run_index\":" << run << ",\"node_id\":"; js(tfout, c.id); tfout << ",\"waypoint_id\":" << c.waypoint << ",\"link\":"; js(tfout, ln); tfout << ",\"link_transform\":"; tf(tfout, state.getGlobalLinkTransform(l)); tfout << ",\"collision_body_transform\":"; tf(tfout, state.getCollisionBodyTransform(l, 0)); tfout << "}\n";
    }
    if (disputed.count(c.id)) {
      run_acm_case_world(caseout, ctx, backend, "A", c, state, group, *scene, formal, rawout, repo_root, parts_dir);
      auto b1 = formal; b1.setEntry("forearm_link", "wrist2_link", true); run_acm_case_world(caseout, ctx, backend, "B1", c, state, group, *scene, b1, rawout, repo_root, parts_dir);
      auto b2 = formal; b2.setEntry("forearm_link", "wrist3_link", true); run_acm_case_world(caseout, ctx, backend, "B2", c, state, group, *scene, b2, rawout, repo_root, parts_dir);
      auto b3 = formal; b3.setEntry("forearm_link", "wrist2_link", true); b3.setEntry("forearm_link", "wrist3_link", true); run_acm_case_world(caseout, ctx, backend, "B3", c, state, group, *scene, b3, rawout, repo_root, parts_dir);
      auto c1 = formal; for (auto* l1 : model->getLinkModels()) for (auto* l2 : model->getLinkModels()) if (l1 < l2) c1.setEntry(l1->getName(), l2->getName(), true); c1.setEntry("forearm_link", "wrist2_link", false); run_acm_case(caseout, ctx, backend, "C1", c, state, group, *scene, c1, rawout, repo_root, parts_dir);
      auto c2 = formal; for (auto* l1 : model->getLinkModels()) for (auto* l2 : model->getLinkModels()) if (l1 < l2) c2.setEntry(l1->getName(), l2->getName(), true); c2.setEntry("forearm_link", "wrist3_link", false); run_acm_case(caseout, ctx, backend, "C2", c, state, group, *scene, c2, rawout, repo_root, parts_dir);
    }
  }
  if (dispatcher) dispatcher->setNearCallback(previous);
  if (backend == "bullet") gContactAddedCallback = previous_contact_added;
  prov.close(); rclcpp::shutdown(); return 0;
}
