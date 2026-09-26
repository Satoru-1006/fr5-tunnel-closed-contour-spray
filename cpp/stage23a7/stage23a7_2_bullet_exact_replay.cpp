// Isolated Bullet-only replay for Stage 2.3A.7.2.
// The input is generated from captured runtime shape vertices and transforms.
// This executable never loads or changes the formal MoveIt model.
#include <btBulletCollisionCommon.h>
#include <BulletCollision/CollisionShapes/btBvhTriangleMeshShape.h>
#include <BulletCollision/CollisionShapes/btTriangleMesh.h>
#include <BulletCollision/CollisionDispatch/btCollisionObjectWrapper.h>
#include <BulletCollision/CollisionDispatch/btManifoldResult.h>

#include <cxxabi.h>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <typeinfo>
#include <vector>

struct Row {
  std::string candidate;
  std::string waypoint;
  std::string pair;
  float runtime_min = std::numeric_limits<float>::quiet_NaN();
  int runtime_contacts = 0;
  std::string vertex_file;
  std::uint64_t v0_offset = 0, v1_offset = 0;
  int v0_count = 0, v1_count = 0;
  float m0 = 0, m1 = 0;
  btVector3 s0{1, 1, 1}, s1{1, 1, 1};
  btTransform t0, t1;
  std::string tri0_file, tri1_file;
  int tri0_count = 0, tri1_count = 0;
  std::string runtime_algorithm;
};

struct Contact {
  btScalar distance = 0;
  btVector3 normal{0, 0, 0};
  btVector3 a{0, 0, 0};
  btVector3 b{0, 0, 0};
  int part0 = -1, part1 = -1, index0 = -1, index1 = -1;
};

struct Capture : btCollisionWorld::ContactResultCallback {
  std::vector<Contact> contacts;
  btScalar addSingleResult(btManifoldPoint& cp,
                           const btCollisionObjectWrapper* obj0, int part0, int index0,
                           const btCollisionObjectWrapper* obj1, int part1, int index1) override {
    Contact c;
    c.distance = cp.m_distance1;
    c.normal = cp.m_normalWorldOnB;
    c.a = cp.m_positionWorldOnA;
    c.b = cp.m_positionWorldOnB;
    c.part0 = part0; c.part1 = part1; c.index0 = index0; c.index1 = index1;
    contacts.push_back(c);
    return cp.m_distance1;
  }
};

static std::vector<std::string> split_tab(const std::string& line) {
  std::vector<std::string> out;
  std::stringstream ss(line);
  std::string item;
  while (std::getline(ss, item, '\t')) out.push_back(item);
  if (!out.empty() && !out.back().empty() && out.back().back() == '\r') out.back().pop_back();
  return out;
}

static double d(const std::vector<std::string>& v, std::size_t i) {
  return std::stod(v.at(i));
}

static int i(const std::vector<std::string>& v, std::size_t n) {
  return std::stoi(v.at(n));
}

static btTransform transform(const std::vector<std::string>& v, std::size_t p) {
  btMatrix3x3 r(static_cast<btScalar>(d(v, p + 0)), static_cast<btScalar>(d(v, p + 1)), static_cast<btScalar>(d(v, p + 2)),
                static_cast<btScalar>(d(v, p + 4)), static_cast<btScalar>(d(v, p + 5)), static_cast<btScalar>(d(v, p + 6)),
                static_cast<btScalar>(d(v, p + 8)), static_cast<btScalar>(d(v, p + 9)), static_cast<btScalar>(d(v, p + 10)));
  return btTransform(r, btVector3(static_cast<btScalar>(d(v, p + 3)), static_cast<btScalar>(d(v, p + 7)), static_cast<btScalar>(d(v, p + 11))));
}

static Row parse_row(const std::vector<std::string>& v) {
  // candidate waypoint pair runtime_min runtime_contact_count vertex_file
  // v0_offset v0_count m0 sx sy sz t0[12] v1_offset v1_count m1 sx sy sz t1[12]
  // tri0_file tri0_count tri1_file tri1_count
  Row r;
  r.candidate = v.at(0); r.waypoint = v.at(1); r.pair = v.at(2);
  r.runtime_min = static_cast<float>(d(v, 3)); r.runtime_contacts = i(v, 4); r.vertex_file = v.at(5);
  r.v0_offset = static_cast<std::uint64_t>(std::stoull(v.at(6))); r.v0_count = i(v, 7); r.m0 = static_cast<float>(d(v, 8));
  r.s0 = btVector3(static_cast<btScalar>(d(v, 9)), static_cast<btScalar>(d(v, 10)), static_cast<btScalar>(d(v, 11))); r.t0 = transform(v, 12);
  r.v1_offset = static_cast<std::uint64_t>(std::stoull(v.at(24))); r.v1_count = i(v, 25); r.m1 = static_cast<float>(d(v, 26));
  r.s1 = btVector3(static_cast<btScalar>(d(v, 27)), static_cast<btScalar>(d(v, 28)), static_cast<btScalar>(d(v, 29))); r.t1 = transform(v, 30);
  r.tri0_file = v.at(42); r.tri0_count = i(v, 43); r.tri1_file = v.at(44); r.tri1_count = i(v, 45);
  if (v.size() > 46) r.runtime_algorithm = v.at(46);
  return r;
}

static std::vector<btVector3> read_vertices(const std::string& path, std::uint64_t offset, int count) {
  std::ifstream f(path, std::ios::binary);
  f.seekg(static_cast<std::streamoff>(offset));
  std::vector<btVector3> out;
  for (int n = 0; n < count; ++n) {
    float xyz[3]{};
    f.read(reinterpret_cast<char*>(xyz), sizeof(xyz));
    if (!f) throw std::runtime_error("vertex file read failed: " + path);
    out.emplace_back(static_cast<btScalar>(xyz[0]), static_cast<btScalar>(xyz[1]), static_cast<btScalar>(xyz[2]));
  }
  return out;
}

static std::vector<btVector3> read_triangles(const std::string& path, int count) {
  std::ifstream f(path, std::ios::binary);
  std::vector<btVector3> out;
  for (int n = 0; n < count * 3; ++n) {
    float xyz[3]{};
    f.read(reinterpret_cast<char*>(xyz), sizeof(xyz));
    if (!f) throw std::runtime_error("triangle file read failed: " + path);
    out.emplace_back(static_cast<btScalar>(xyz[0]), static_cast<btScalar>(xyz[1]), static_cast<btScalar>(xyz[2]));
  }
  return out;
}

struct ReplayResult {
  std::vector<Contact> contacts;
  std::string algorithm;
};

static std::string demangle(const char* name) {
  int status = 0;
  char* raw = abi::__cxa_demangle(name, nullptr, nullptr, &status);
  std::string result = (status == 0 && raw) ? raw : name;
  std::free(raw);
  return result;
}

static ReplayResult pair_test(btCollisionShape* a_shape, const btTransform& a_tf,
                              btCollisionShape* b_shape, const btTransform& b_tf) {
  btDefaultCollisionConfiguration config;
  btCollisionDispatcher dispatcher(&config);
  btCollisionObject a, b;
  a.setCollisionShape(a_shape); b.setCollisionShape(b_shape);
  a.setWorldTransform(a_tf); b.setWorldTransform(b_tf);
  btCollisionObjectWrapper wa(nullptr, a_shape, &a, a_tf, -1, -1);
  btCollisionObjectWrapper wb(nullptr, b_shape, &b, b_tf, -1, -1);
  btCollisionAlgorithm* algorithm = dispatcher.findAlgorithm(&wa, &wb, nullptr, BT_CLOSEST_POINT_ALGORITHMS);
  ReplayResult result;
  result.algorithm = algorithm ? demangle(typeid(*algorithm).name()) : "not_available_no_algorithm";
  btDispatcherInfo info;
  info.m_dispatchFunc = btDispatcherInfo::DISPATCH_DISCRETE;
  btManifoldResult manifold_result(&wa, &wb);
  if (algorithm) {
    algorithm->processCollision(&wa, &wb, info, &manifold_result);
    if (const btPersistentManifold* manifold = manifold_result.getPersistentManifold()) {
      for (int n = 0; n < manifold->getNumContacts(); ++n) {
        const btManifoldPoint& cp = manifold->getContactPoint(n);
        Contact c;
        c.distance = cp.m_distance1;
        c.normal = cp.m_normalWorldOnB;
        c.a = cp.m_positionWorldOnA;
        c.b = cp.m_positionWorldOnB;
        c.part0 = cp.m_partId0; c.part1 = cp.m_partId1; c.index0 = cp.m_index0; c.index1 = cp.m_index1;
        result.contacts.push_back(c);
      }
    }
    dispatcher.freeCollisionAlgorithm(algorithm);
  }
  return result;
}

static double min_distance(const std::vector<Contact>& c) {
  if (c.empty()) return std::numeric_limits<double>::quiet_NaN();
  double x = std::numeric_limits<double>::infinity();
  for (const auto& p : c) x = std::min(x, static_cast<double>(p.distance));
  return x;
}

static std::string sign(double x) {
  if (!std::isfinite(x)) return "not_available";
  return std::signbit(x) ? "negative" : "nonnegative";
}

static void write_contacts(std::ofstream& out, const Row& r, const char* group, const std::vector<Contact>& c) {
  for (std::size_t n = 0; n < c.size(); ++n) {
    const auto& p = c[n];
    out << r.candidate << ',' << r.waypoint << ',' << group << ',' << n << ','
        << std::setprecision(17) << static_cast<double>(p.distance) << ','
        << static_cast<double>(p.normal.x()) << ',' << static_cast<double>(p.normal.y()) << ',' << static_cast<double>(p.normal.z()) << ','
        << static_cast<double>(p.a.x()) << ',' << static_cast<double>(p.a.y()) << ',' << static_cast<double>(p.a.z()) << ','
        << static_cast<double>(p.b.x()) << ',' << static_cast<double>(p.b.y()) << ',' << static_cast<double>(p.b.z()) << ','
        << p.part0 << ',' << p.part1 << ',' << p.index0 << ',' << p.index1 << '\n';
  }
}

static void write_result(std::ofstream& out, const Row& r, const char* group,
                         const ReplayResult& replay_result, bool shape_match) {
  const auto& c = replay_result.contacts;
  const double replay = min_distance(c), runtime = static_cast<double>(r.runtime_min);
  const bool finite = std::isfinite(runtime) && std::isfinite(replay);
  const double abs_error = finite ? std::abs(runtime - replay) : std::numeric_limits<double>::quiet_NaN();
  const double rel_error = finite && runtime != 0 ? abs_error / std::abs(runtime) : std::numeric_limits<double>::quiet_NaN();
  const bool sign_match = finite && (std::signbit(runtime) == std::signbit(replay));
  const bool algorithm_match = group[0] != 'D' && !r.runtime_algorithm.empty() && r.runtime_algorithm == replay_result.algorithm;
  out << r.candidate << ',' << r.waypoint << ',' << r.pair << ',' << group << ','
      << std::setprecision(17) << runtime << ',' << replay << ',' << abs_error << ',' << rel_error << ','
      << sign(runtime) << ',' << sign(replay) << ',' << (sign_match ? "true" : "false") << ','
      << r.runtime_contacts << ',' << c.size() << ',' << (shape_match ? "true" : "false") << ','
      << r.runtime_algorithm << ',' << replay_result.algorithm << ',' << (algorithm_match ? "true" : "false") << ','
      << ((!c.empty() && replay <= 0) ? "true" : "false") << '\n';
}

int main(int argc, char** argv) {
  if (argc != 5 && argc != 6) { std::cerr << "usage: replay INPUT.tsv RESULTS.csv CONTACTS.csv RUN_ID [GROUP]\n"; return 2; }
  const std::string group_mode = argc == 6 ? argv[5] : "ALL";
  std::ifstream input(argv[1]);
  std::ofstream results(argv[2]);
  std::ofstream contacts(argv[3]);
  if (!input || !results || !contacts) return 3;
  results << "candidate_id,waypoint_id,pair,group,runtime_m_distance1,replay_m_distance1,absolute_error,relative_error,runtime_sign,replay_sign,sign_match,contact_count_runtime,contact_count_replay,leaf_pair_match,runtime_algorithm_type,replay_algorithm_type,algorithm_match,replay_collision\n";
  contacts << "candidate_id,waypoint_id,group,contact_index,m_distance1,normal_x,normal_y,normal_z,pos_a_x,pos_a_y,pos_a_z,pos_b_x,pos_b_y,pos_b_z,part_id_0,part_id_1,index_0,index_1\n";
  std::string line;
  std::size_t row_count = 0;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#') continue;
    const auto fields = split_tab(line);
    if (fields.size() < 46 || fields[0] == "candidate_id") continue;
    const Row r = parse_row(fields);
    auto v0 = read_vertices(r.vertex_file, r.v0_offset, r.v0_count);
    auto v1 = read_vertices(r.vertex_file, r.v1_offset, r.v1_count);
    btConvexHullShape a(v0.empty() ? nullptr : &v0[0].x(), static_cast<int>(v0.size()), sizeof(btVector3));
    btConvexHullShape b(v1.empty() ? nullptr : &v1[0].x(), static_cast<int>(v1.size()), sizeof(btVector3));
    a.setLocalScaling(r.s0); b.setLocalScaling(r.s1); a.setMargin(r.m0); b.setMargin(r.m1);
    if (group_mode == "C") {
      auto cc = pair_test(&a, r.t0, &b, r.t1);
      write_contacts(contacts, r, "C", cc.contacts); write_result(results, r, "C", cc, true);
      ++row_count;
      continue;
    }
    auto ca = pair_test(&a, r.t0, &b, r.t1);
    write_contacts(contacts, r, "A", ca.contacts); write_result(results, r, "A", ca, true);
    a.setMargin(0); b.setMargin(0);
    auto cb = pair_test(&a, r.t0, &b, r.t1);
    write_contacts(contacts, r, "B", cb.contacts); write_result(results, r, "B", cb, true);
    auto tri0 = read_triangles(r.tri0_file, r.tri0_count), tri1 = read_triangles(r.tri1_file, r.tri1_count);
    btTriangleMesh mesh0, mesh1;
    for (int n = 0; n < r.tri0_count; ++n) mesh0.addTriangle(tri0[3*n], tri0[3*n+1], tri0[3*n+2], true);
    for (int n = 0; n < r.tri1_count; ++n) mesh1.addTriangle(tri1[3*n], tri1[3*n+1], tri1[3*n+2], true);
    btBvhTriangleMeshShape d0(&mesh0, true, true), d1(&mesh1, true, true);
    d0.setLocalScaling(r.s0); d1.setLocalScaling(r.s1); d0.setMargin(r.m0); d1.setMargin(r.m1);
    auto cd = pair_test(&d0, r.t0, &d1, r.t1);
    write_contacts(contacts, r, "D", cd.contacts); write_result(results, r, "D", cd, false);
    ++row_count;
  }
  std::cerr << "replay_version=3 replayed_rows=" << row_count << " btScalar_bytes=" << sizeof(btScalar)
            << " double_precision=" << (sizeof(btScalar) == 8 ? "true" : "false") << "\n";
  return 0;
}
