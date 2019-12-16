#ifndef SLAM_CTOR_PARTICLE_FILTER_H_INCLUDED
#define SLAM_CTOR_PARTICLE_FILTER_H_INCLUDED

#include <memory>
#include <vector>
#include <random>
#include <unordered_set>

/* An element of ParticleFilter. Used to approximate target distribution */
class Particle {
public:
  Particle(): _weight(1.0) {}
  virtual ~Particle() = default;

  double weight() const { return _weight; }
  void set_weight(double w) { _weight = w; }

  virtual void sample() {}
private:
  double _weight;
};

template <typename ParticleT>
class ParticleFactory {
public:
  virtual std::shared_ptr<ParticleT> create_particle() = 0;
  virtual ~ParticleFactory() = default;
};

template <typename ParticlePtr>
class UniformResamling { /* Strategy */
public:
  // TODO: should be moved to separate strategy
  bool resampling_is_required(std::vector<ParticlePtr> particles) {
    // TODO: resampling is not required if robot haven't traveled far enough
    // Computer N_eff from gMapping (based on Doucet work)
    double sq_sum = 0;
    for (auto &p : particles) {
      sq_sum += p->weight() * p->weight();
    }
    double effective_particles_cnt = 1.0 / sq_sum;
    return effective_particles_cnt * 2 < particles.size();
  }

  std::vector<unsigned> resample(std::vector<ParticlePtr> particles) {
    // TODO: implement O(n) resampling from gmapping
    //double total_weight = 0;
    //for (auto &p : particles) { total_weight += particles.weight; }
    std::vector<unsigned> new_particle_inds(particles.size());

    std::random_device rd;
    std::mt19937 engine(rd());
    std::uniform_real_distribution<> uniform_distr(0, 1);
    for (uint i = 0; i < particles.size(); i++) {
      double sample = uniform_distr(engine);
      double total_w = 0;
      for (uint j = 0; j < particles.size(); j++) {
        total_w += particles[j]->weight();
        if (sample < total_w) {
          new_particle_inds[i] = j;
          break;
        }
      }
    }
    return new_particle_inds;
  }
};

template <typename ParticleT>
class ParticleFilter {
private:
  using ParticlePtr = std::shared_ptr<ParticleT>;
public: // methods
  ParticleFilter(std::shared_ptr<ParticleFactory<ParticleT>> p_ftry,
                 unsigned n = 1) : _particle_supplier{p_ftry} {
    for (unsigned i = 0; i < n; i++) {
      ParticlePtr particle = p_ftry->create_particle();
      particle->set_weight(1.0 / n);
      _particles.push_back(particle);
    }
  }

  bool try_resample() {
    // TODO: heuristic with traveled distange, move to common sample_particles
    if (!_resampler.resampling_is_required(_particles)) {
      return false;
    }

    std::vector<ParticlePtr> new_particles;
    std::unordered_set<unsigned> new_part_inds;
    for (unsigned i : _resampler.resample(_particles)) {
      ParticlePtr sampled = _particles[i];
      if (new_part_inds.count(i)) {
        ParticlePtr new_particle = _particle_supplier->create_particle();
        *new_particle = *sampled;
        new_particle->sample();
        sampled = new_particle;
      } else {
        new_part_inds.insert(i);
      }
      new_particles.push_back(sampled);
    }
    _particles = std::move(new_particles);
    normalize_weights();
    return true;
  }

  void normalize_weights() {
    double total_weight = 0;
    for (auto &p : _particles) { total_weight += p->weight(); }
    for (auto &p : _particles) { p->set_weight(p->weight() / total_weight); }
  }

  const ParticleT& heaviest_particle() const {
    const ParticleT *heaviest = nullptr;
    for (auto &p : particles()) {
      if (heaviest && p->weight() < heaviest->weight()) { continue; }
      heaviest = p.get();
    }
    return *heaviest;
  }

  ParticleT& heaviest_particle() {
    return const_cast<ParticleT&>(
      static_cast<const ParticleFilter<ParticleT>>(*this).heaviest_particle());
  }

  inline std::vector<ParticlePtr>& particles() { return _particles; }
  inline const std::vector<ParticlePtr>& particles() const {
    return _particles;
  }

private:
  std::shared_ptr<ParticleFactory<ParticleT>> _particle_supplier;
  std::vector<ParticlePtr> _particles;
  /* TODO: make template parameter/set at runtime */
  UniformResamling<ParticlePtr> _resampler;
};

#endif // header-guard
