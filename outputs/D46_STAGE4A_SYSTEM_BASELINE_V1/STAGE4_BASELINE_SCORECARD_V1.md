# STAGE4_BASELINE_SCORECARD_V1

Measurement pipeline: **PASS**.

```json
{
  "accuracy": {
    "by_family_terminal_position": {
      "ADVERSARIAL": {
        "count": 200,
        "finite_count": 200,
        "max": 0.007763962262102868,
        "mean": 0.0006655179879542759,
        "median": 3.745109062575315e-08,
        "min": 3.745109062575315e-08,
        "p90": 0.0005546057109744672,
        "p95": 0.006654844483438186,
        "p99": 0.007763962262102868,
        "status": "AVAILABLE",
        "std": 0.0020117869144054275,
        "unit": "m"
      },
      "BOUNDARY": {
        "count": 200,
        "finite_count": 200,
        "max": 0.46750771743640723,
        "mean": 0.23374684072141613,
        "median": 0.24476917214051008,
        "min": 3.745109062575315e-08,
        "p90": 0.46649417619379474,
        "p95": 0.46732176699864736,
        "p99": 0.46750771743640723,
        "status": "AVAILABLE",
        "std": 0.15264173884022894,
        "unit": "m"
      },
      "COLLISION_SENSITIVE": {
        "count": 200,
        "finite_count": 200,
        "max": 3.745109088418557e-08,
        "mean": 3.745109067958589e-08,
        "median": 3.745109062575315e-08,
        "min": 3.745109062575315e-08,
        "p90": 3.745109087274807e-08,
        "p95": 3.7451090873219e-08,
        "p99": 3.745109088418557e-08,
        "status": "AVAILABLE",
        "std": 9.411866384555965e-17,
        "unit": "m"
      },
      "NORMAL": {
        "count": 200,
        "finite_count": 200,
        "max": 3.745109062575315e-08,
        "mean": 3.745109062575315e-08,
        "median": 3.745109062575315e-08,
        "min": 3.745109062575315e-08,
        "p90": 3.745109062575315e-08,
        "p95": 3.745109062575315e-08,
        "p99": 3.745109062575315e-08,
        "status": "AVAILABLE",
        "std": 0.0,
        "unit": "m"
      },
      "PERTURBATION": {
        "count": 150,
        "finite_count": 150,
        "max": 0.001899340887355807,
        "mean": 0.0005573536708239756,
        "median": 0.000474665887114224,
        "min": 6.296943796069699e-05,
        "p90": 0.0009858607375878544,
        "p95": 0.0012409706827403578,
        "p99": 0.0014925317523508854,
        "status": "AVAILABLE",
        "std": 0.0003193488413651112,
        "unit": "m"
      },
      "REGRESSION": {
        "count": 50,
        "finite_count": 50,
        "max": 3.745109062575315e-08,
        "mean": 3.745109062575315e-08,
        "median": 3.745109062575315e-08,
        "min": 3.745109062575315e-08,
        "p90": 3.745109062575315e-08,
        "p95": 3.745109062575315e-08,
        "p99": 3.745109062575315e-08,
        "statistical_note": "small_sample_tail_percentiles_are_descriptive_only",
        "status": "AVAILABLE",
        "std": 0.0,
        "unit": "m"
      }
    },
    "tcp_trajectory_error_max_m": {
      "count": 1000,
      "finite_count": 1000,
      "max": 1.021263622966719,
      "mean": 0.24093530485979664,
      "median": 0.0686862019029078,
      "min": 0.06288239058815649,
      "p90": 0.7417162834513457,
      "p95": 0.8875172610850125,
      "p99": 0.9917251706700924,
      "status": "AVAILABLE",
      "std": 0.2829961979514252,
      "unit": "m"
    },
    "tcp_trajectory_error_p95_m": {
      "count": 1000,
      "finite_count": 1000,
      "max": 0.969534456828696,
      "mean": 0.2105833352044893,
      "median": 0.026787857174056198,
      "min": 0.02220621383586147,
      "p90": 0.7305936690045943,
      "p95": 0.866091878266764,
      "p99": 0.9568977796092227,
      "status": "AVAILABLE",
      "std": 0.29189818446259436,
      "unit": "m"
    },
    "tcp_trajectory_error_p99_m": {
      "count": 1000,
      "finite_count": 1000,
      "max": 1.018573389824559,
      "mean": 0.23460347313824592,
      "median": 0.059420665835148975,
      "min": 0.054186374850330246,
      "p90": 0.7409092911948614,
      "p95": 0.8859009478440284,
      "p99": 0.9891931609439427,
      "status": "AVAILABLE",
      "std": 0.2861712348548857,
      "unit": "m"
    },
    "terminal_orientation_error_rad": {
      "count": 1000,
      "finite_count": 1000,
      "max": 3.033831376902675,
      "mean": 1.9168923308185444,
      "median": 1.8406330601230243,
      "min": 1.4074075821960317,
      "p90": 2.3192803182506188,
      "p95": 2.502785052997149,
      "p99": 3.0308112711989166,
      "status": "AVAILABLE",
      "std": 0.2722212032720847,
      "unit": "rad"
    },
    "terminal_position_error_m": {
      "count": 1000,
      "finite_count": 1000,
      "max": 0.46750771743640723,
      "mean": 0.04696609164548848,
      "median": 3.745109062575315e-08,
      "min": 3.745109062575315e-08,
      "p90": 0.24476917214051008,
      "p95": 0.361456919043345,
      "p99": 0.46732176699864736,
      "status": "AVAILABLE",
      "std": 0.11568302972522089,
      "unit": "m"
    }
  },
  "benchmark_id": "STAGE4_SYSTEM_BENCHMARK_V1",
  "coverage_gaps": [
    "model-based required torque UNAVAILABLE",
    "continuous self-collision NOT_AVAILABLE",
    "physical TCP calibration uncertainty UNAVAILABLE",
    "cross-platform bitwise determinism not claimed",
    "jerk and effort provenance not vendor-certified"
  ],
  "dynamics": {
    "model_based_required_torque_status": "UNAVAILABLE",
    "raw_torque_values": null,
    "reason": "No Pinocchio or validated project inverse-dynamics backend installed; URDF effort values are provenance only",
    "torque_limit_ratio": null,
    "torque_margin": null
  },
  "family_counts": {
    "ADVERSARIAL": 200,
    "BOUNDARY": 200,
    "COLLISION_SENSITIVE": 200,
    "NORMAL": 200,
    "PERTURBATION": 150,
    "REGRESSION": 50
  },
  "frozen_robot_baseline_performance_status": "MEASURED_WITH_FAILURES_AND_COVERAGE_LIMITS",
  "geometry_collision": {
    "clearance_threshold_status": "UNRESOLVED_THRESHOLD",
    "collision_method": "adaptive_discrete_interpolation",
    "continuous_self_collision_status": "NOT_AVAILABLE",
    "dense_environment_collision_samples": 121900,
    "dense_self_collision_samples": 85021,
    "environment_ccd_failures": 15687,
    "environment_ccd_status": "AVAILABLE_NATIVE_BULLET_ROBOT_WORLD",
    "environment_clearance_m": {
      "count": 1000,
      "finite_count": 1000,
      "max": 0.09923500314732128,
      "mean": -0.01676783860062103,
      "median": 0.08857717027025891,
      "min": -0.5925596612628776,
      "p90": 0.08979578098767822,
      "p95": 0.09256145596759142,
      "p99": 0.09914139248778407,
      "status": "AVAILABLE",
      "std": 0.21101056492627832,
      "unit": "m"
    },
    "self_clearance_m": {
      "count": 1000,
      "finite_count": 1000,
      "max": 0.016627569390579144,
      "mean": 0.00446304498750776,
      "median": 0.016623586820841103,
      "min": -0.08030684992908682,
      "p90": 0.016624860786399655,
      "p95": 0.016626126098167895,
      "p99": 0.016627569390579144,
      "status": "AVAILABLE",
      "std": 0.02802304720403193,
      "unit": "m"
    },
    "waypoint_environment_collision_cases": 214,
    "waypoint_self_collision_cases": 203
  },
  "kinematics_motion_quality": {
    "acceleration_limit_violations": 0,
    "acceleration_ratio": {
      "count": 1000,
      "finite_count": 1000,
      "max": 0.3072575255263954,
      "mean": 0.04378945735608042,
      "median": 0.014284396548881468,
      "min": 0.006646802174805208,
      "p90": 0.12281516191285899,
      "p95": 0.19850895039666885,
      "p99": 0.30209667265482804,
      "status": "AVAILABLE",
      "std": 0.06344921139119475,
      "unit": "ratio"
    },
    "continuity_position_failures": 0,
    "derivative_method": "native_Ruckig_exported_states_for_actual_post_Ruckig_replay; numpy_gradient_for_declared audit-only perturbations",
    "jerk_limit_violations": 0,
    "jerk_ratio": {
      "count": 1000,
      "finite_count": 1000,
      "max": 0.035685059175868006,
      "mean": 0.0032397328377178776,
      "median": 0.0003898243482241949,
      "min": 0.00012042276585485603,
      "p90": 0.00971987558804125,
      "p95": 0.01954959478731813,
      "p99": 0.035678667108282394,
      "status": "AVAILABLE",
      "std": 0.006960050961852593,
      "unit": "ratio"
    },
    "joint_limit_violations": 0,
    "max_discontinuity": {
      "acceleration_rad_s2": {
        "count": 1000,
        "finite_count": 1000,
        "max": 0.1389016802423416,
        "mean": 0.020308526439914792,
        "median": 0.006525176730393311,
        "min": 0.0031259765308737086,
        "p90": 0.05838561373860894,
        "p95": 0.09304236405691808,
        "p99": 0.13882489595226116,
        "status": "AVAILABLE",
        "std": 0.029566203094965233,
        "unit": "rad/s^2"
      },
      "position_rad": {
        "count": 1000,
        "finite_count": 1000,
        "max": 0.2833091226820392,
        "mean": 0.2629412611525771,
        "median": 0.26165610209215023,
        "min": 0.26002499243987753,
        "p90": 0.26763156312478126,
        "p95": 0.27006042874731545,
        "p99": 0.2739578774999244,
        "status": "AVAILABLE",
        "std": 0.003047016581782566,
        "unit": "rad"
      },
      "velocity_rad_s": {
        "count": 1000,
        "finite_count": 1000,
        "max": 0.2947462528573131,
        "mean": 0.09214793706038242,
        "median": 0.06264965823229895,
        "min": 0.04259776893762631,
        "p90": 0.19464503246440262,
        "p95": 0.2426898549846721,
        "p99": 0.2921278444180672,
        "status": "AVAILABLE",
        "std": 0.06236950939736433,
        "unit": "rad/s"
      }
    },
    "min_joint_limit_margin_rad": {
      "count": 1000,
      "finite_count": 1000,
      "max": 0.1903926615353102,
      "mean": 0.013875706713356936,
      "median": 0.00010000000000021103,
      "min": 1.000000000139778e-06,
      "p90": 0.012986852055916527,
      "p95": 0.1903926615353102,
      "p99": 0.1903926615353102,
      "status": "AVAILABLE",
      "std": 0.046988807110496744,
      "unit": "rad"
    },
    "velocity_limit_violations": 0,
    "velocity_ratio": {
      "count": 1000,
      "finite_count": 1000,
      "max": 0.09179183263909292,
      "mean": 0.02909550387441036,
      "median": 0.019671974233567657,
      "min": 0.013680406926778712,
      "p90": 0.059016723681615396,
      "p95": 0.07431989185421085,
      "p99": 0.09155879424962834,
      "status": "AVAILABLE",
      "std": 0.019028032085647652,
      "unit": "ratio"
    }
  },
  "legacy_stage3_health": {
    "D41_strict_replay_status": "PASS",
    "H1": 7.089328839013259e-05,
    "H32": 0.01849100619381173,
    "canonical_checkpoint_sha256": "37645eea5aef8ce2fe7084c83f35429a7a13fbe3e9612eb84417ccce6cba9742",
    "canonical_update": 480,
    "stage3_release_status": "FROZEN_AND_CLOSED"
  },
  "measurement_pipeline_status": "PASS",
  "post_ruckig_comparison": {
    "metrics": {
      "max_acceleration_ratio": {
        "delta_post_minus_pre": -1.0398452932047775e-13,
        "post_ruckig": 0.009579396534353324,
        "pre_ruckig": 0.009579396534457309
      },
      "max_condition_number": {
        "delta_post_minus_pre": 0.0,
        "post_ruckig": 11822852191.641027,
        "pre_ruckig": 11822852191.641027
      },
      "max_jerk_ratio": {
        "delta_post_minus_pre": 2.2299429554780944e-05,
        "post_ruckig": 0.00023042942821655058,
        "pre_ruckig": 0.00020812999866176964
      },
      "max_velocity_ratio": {
        "delta_post_minus_pre": -9.852466065218835e-13,
        "post_ruckig": 0.016421107188169345,
        "pre_ruckig": 0.016421107189154592
      },
      "min_environment_clearance_m": {
        "delta_post_minus_pre": 0.0,
        "post_ruckig": 0.08857717027025891,
        "pre_ruckig": 0.08857717027025891
      },
      "min_self_clearance_m": {
        "delta_post_minus_pre": 0.0,
        "post_ruckig": 0.016623586820841103,
        "pre_ruckig": 0.016623586820841103
      },
      "min_sigma_min": {
        "delta_post_minus_pre": 0.0,
        "post_ruckig": 1.5720330432378874e-10,
        "pre_ruckig": 1.5720330432378874e-10
      },
      "native_waypoint_environment_collision_count": {
        "delta_post_minus_pre": 0.0,
        "post_ruckig": 0,
        "pre_ruckig": 0
      },
      "native_waypoint_self_collision_count": {
        "delta_post_minus_pre": 0.0,
        "post_ruckig": 0,
        "pre_ruckig": 0
      },
      "tcp_trajectory_error_max_m": {
        "delta_post_minus_pre": 0.0,
        "post_ruckig": 0.06852261993399261,
        "pre_ruckig": 0.06852261993399261
      },
      "terminal_orientation_error_rad": {
        "delta_post_minus_pre": 0.0,
        "post_ruckig": 1.8406330601230243,
        "pre_ruckig": 1.8406330601230243
      },
      "terminal_position_error_m": {
        "delta_post_minus_pre": 0.0,
        "post_ruckig": 3.745109062575315e-08,
        "pre_ruckig": 3.745109062575315e-08
      },
      "trajectory_duration_s": {
        "delta_post_minus_pre": 0.0,
        "post_ruckig": 562.068413668,
        "pre_ruckig": 562.068413668
      }
    },
    "post_case_id": "regression_0000",
    "pre_case_id": "regression_0001",
    "status": "AVAILABLE"
  },
  "reproducibility": {
    "checkpoint_reload": "PASS_D45_AUTHENTICATED_STAGE3_REPLAY",
    "clean_restart": "PENDING_FROM_REGRESSION_CASES",
    "fixed_seed_case_identity": "PASS",
    "fresh_process_replay": "PENDING_FROM_REGRESSION_CASES",
    "same_process_replay": "PENDING_FROM_REGRESSION_CASES"
  },
  "robustness": {
    "ADVERSARIAL": {
      "audit_only_cases": 200,
      "cases": 200,
      "collision_failures": 0,
      "finite_failures": 0,
      "joint_limit_failures": 0
    },
    "BOUNDARY": {
      "audit_only_cases": 200,
      "cases": 200,
      "collision_failures": 100,
      "finite_failures": 0,
      "joint_limit_failures": 0
    },
    "COLLISION_SENSITIVE": {
      "audit_only_cases": 200,
      "cases": 200,
      "collision_failures": 196,
      "finite_failures": 0,
      "joint_limit_failures": 0
    },
    "NORMAL": {
      "audit_only_cases": 200,
      "cases": 200,
      "collision_failures": 0,
      "finite_failures": 0,
      "joint_limit_failures": 0
    },
    "PERTURBATION": {
      "audit_only_cases": 150,
      "cases": 150,
      "collision_failures": 0,
      "finite_failures": 0,
      "joint_limit_failures": 0
    },
    "REGRESSION": {
      "audit_only_cases": 1,
      "cases": 50,
      "collision_failures": 0,
      "finite_failures": 0,
      "joint_limit_failures": 0
    }
  },
  "ruckig": {
    "applicable_cases": 49,
    "scope": "actual frozen D41 post-Ruckig replay controls only; audit-only stress and perturbation cases are not relabelled as generated Ruckig trajectories",
    "valid_cases": 49,
    "validation_failures": 0,
    "validity_rate": 1.0
  },
  "schema_version": "stage4-baseline-scorecard-v1",
  "scope": "Stage 0/1 ON-state open-arch only",
  "scorecard_id": "STAGE4_BASELINE_SCORECARD_V1",
  "singularity": {
    "condition_number": {
      "count": 1000,
      "finite_count": 1000,
      "max": 149612347097.48575,
      "mean": 6857071747.376625,
      "median": 26908.564889061403,
      "min": 68.29514576431946,
      "p90": 11822852053.110065,
      "p95": 11822852213.446491,
      "p99": 148126459937.77783,
      "status": "AVAILABLE",
      "std": 25703814223.20038,
      "unit": "ratio"
    },
    "measurement": "native MoveIt2 RobotState Jacobian SVD",
    "minimum_singular_value": {
      "count": 1000,
      "finite_count": 1000,
      "max": 0.026950735027824062,
      "mean": 0.0017904028939406303,
      "median": 7.331856733573075e-05,
      "min": 1.3635695874938152e-11,
      "p90": 0.0028505018657188513,
      "p95": 0.012403086244397806,
      "p99": 0.026873378514944774,
      "status": "AVAILABLE",
      "std": 0.005146224706670978,
      "unit": "Jacobian singular-value units"
    },
    "threshold_status": "UNRESOLVED_THRESHOLD"
  },
  "temporal_quality": {
    "actual_post_ruckig_duration_s": 562.068413668,
    "duration_s": {
      "count": 1000,
      "finite_count": 1000,
      "max": 674.4820964016001,
      "mean": 426.14902987480417,
      "median": 483.37883575448,
      "min": 101.17231446023999,
      "p90": 578.93046607804,
      "p95": 590.1718343514,
      "p99": 657.6200439915599,
      "status": "AVAILABLE",
      "std": 161.0675544348769,
      "unit": "s"
    },
    "time_scale_factors": {
      "count": 1000,
      "finite_count": 1000,
      "max": 1.2000000000000002,
      "mean": 0.75818,
      "median": 0.8600000000000001,
      "min": 0.18,
      "p90": 1.03,
      "p95": 1.05,
      "p99": 1.17,
      "status": "AVAILABLE",
      "std": 0.28656218801509736,
      "unit": "ratio"
    }
  },
  "total_cases": 1000,
  "unresolved_thresholds": [
    "environment clearance",
    "self clearance",
    "physical singularity risk",
    "model-based torque acceptance",
    "v/a continuity hard threshold for numerical derivative cases"
  ]
}
```
