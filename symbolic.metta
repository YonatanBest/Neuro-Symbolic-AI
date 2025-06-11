; Initialize knowledge base
!(bind! &kb (new-space))

; Define cast functions between Nat and Number (needed for chaining)
(= (fromNumber $n) (if (<= $n 0) Z (S (fromNumber (- $n 1)))))

; Base case for forward chaining
(= (fcc $kb $_ (: $prf $prms)) (: $prf $prms))

; Recursive step for forward chaining
(= (fcc $kb (S $k) (: $prfarg $prms))
   (let (: $prfabs (-> $prms $ccln)) (bc $kb $k (: $prfabs (-> $prms $ccln)))
     (fcc $kb $k (: ($prfabs $prfarg) $ccln))))

; Base case for backward chaining
(= (bc $kb $_ (: $prf $ccln)) (match $kb (: $prf $ccln) (: $prf $ccln)))

; Recursive step for backward chaining
(= (bc $kb (S $k) (: ($prfabs $prfarg) $ccln))
   (let* (((: $prfabs (-> $prms $ccln)) (bc $kb $k (: $prfabs (-> $prms $ccln))))
          ((: $prfarg $prms) (bc $kb $k (: $prfarg $prms))))
     (: ($prfabs $prfarg) $ccln)))

; Disease Rules
!(add-reduct &kb (: flu-rule 
  (-> (Evaluation has_fever $x)
      (-> (Evaluation has_cough $x)
          (-> (Evaluation has_fatigue $x)
              (Inheritance $x flu))))))

!(add-reduct &kb (: covid-rule 
  (-> (Evaluation has_fever $x)
      (-> (Evaluation has_cough $x)
          (-> (Evaluation has_shortness_of_breath $x)
              (-> (Evaluation has_fatigue $x)
                  (Inheritance $x covid_19)))))))

!(add-reduct &kb (: cold-rule 
  (-> (Evaluation has_runny_nose $x)
      (-> (Evaluation has_cough $x)
          (-> (Evaluation has_sore_throat $x)
              (Inheritance $x common_cold))))))

!(add-reduct &kb (: common-cold-rule
  (-> (Evaluation has_sneezing $x)
      (-> (Evaluation has_runny_nose $x)
          (-> (Evaluation has_sore_throat $x)
              (-> (Evaluation has_cough $x)
                  (-> (Evaluation has_fatigue $x)
                      (-> (Evaluation has_mild_fever $x)
                          (Inheritance $x common_cold)))))))))

!(add-reduct &kb (: asthma-rule
  (-> (Evaluation has_shortness_of_breath $x)
      (-> (Evaluation has_wheezing $x)
          (-> (Evaluation has_cough $x)
              (-> (Evaluation has_chest_tightness $x)
                  (Inheritance $x asthma)))))))

!(add-reduct &kb (: diabetes-rule
  (-> (Evaluation has_frequent_urination $x)
      (-> (Evaluation has_increased_thirst $x)
          (-> (Evaluation has_fatigue $x)
              (-> (Evaluation has_blurred_vision $x)
                  (Inheritance $x diabetes)))))))

!(add-reduct &kb (: hypertension-rule
  (-> (Evaluation has_headache $x)
      (-> (Evaluation has_dizziness $x)
          (-> (Evaluation has_nosebleeds $x)
              (Inheritance $x hypertension))))))

!(add-reduct &kb (: acid-reflux-rule
  (-> (Evaluation has_heartburn $x)
      (-> (Evaluation has_sour_taste $x)
          (-> (Evaluation has_chest_discomfort $x)
              (Inheritance $x acid_reflux))))))

!(add-reduct &kb (: insomnia-rule
  (-> (Evaluation has_difficulty_falling_asleep $x)
      (-> (Evaluation has_difficulty_staying_asleep $x)
          (Inheritance $x insomnia)))))

!(add-reduct &kb (: depression-rule
  (-> (Evaluation has_persistent_sadness $x)
      (-> (Evaluation has_loss_of_interest $x)
          (-> (Evaluation has_fatigue $x)
              (-> (Evaluation has_sleep_issues $x)
                  (Inheritance $x depression)))))))

!(add-reduct &kb (: anxiety-rule
  (-> (Evaluation has_worry $x)
      (-> (Evaluation has_nervousness $x)
          (-> (Evaluation has_restlessness $x)
              (-> (Evaluation has_rapid_heartbeat $x)
                  (Inheritance $x anxiety)))))))

!(add-reduct &kb (: arthritis-rule
  (-> (Evaluation has_joint_pain $x)
      (-> (Evaluation has_stiffness $x)
          (-> (Evaluation has_reduced_mobility $x)
              (Inheritance $x arthritis))))))

!(add-reduct &kb (: ibs-rule
  (-> (Evaluation has_bloating $x)
      (-> (Evaluation has_gas $x)
          (-> (Evaluation has_abdominal_pain $x)
              (-> (Evaluation has_irregular_bowel $x)
                  (Inheritance $x ibs)))))))

!(add-reduct &kb (: migraine-rule
  (-> (Evaluation has_throbbing_headache $x)
      (-> (Evaluation has_nausea $x)
          (-> (Evaluation has_light_sensitivity $x)
              (-> (Evaluation has_sound_sensitivity $x)
                  (Inheritance $x migraine)))))))

!(add-reduct &kb (: allergy-rule
  (-> (Evaluation has_sneezing $x)
      (-> (Evaluation has_itchy_eyes $x)
          (-> (Evaluation has_congestion $x)
              (Inheritance $x seasonal_allergies))))))

!(add-reduct &kb (: constipation-rule
  (-> (Evaluation has_infrequent_stools $x)
      (-> (Evaluation has_hard_stools $x)
          (-> (Evaluation has_bloating $x)
              (Inheritance $x constipation))))))

!(add-reduct &kb (: eczema-rule
  (-> (Evaluation has_dry_skin $x)
      (-> (Evaluation has_itchy_skin $x)
          (-> (Evaluation has_inflamed_skin $x)
              (Inheritance $x eczema))))))

!(add-reduct &kb (: back-pain-rule
  (-> (Evaluation has_back_ache $x)
      (-> (Evaluation has_sharp_back_pain $x)
          (Inheritance $x back_pain)))))

; Treatment Rules
!(add-reduct &kb (: treat-flu 
  (-> (Inheritance $x flu)
      (Evaluation recommend_rest_and_fluids $x))))

!(add-reduct &kb (: treat-covid 
  (-> (Inheritance $x covid_19)
      (Evaluation recommend_isolation_and_testing $x))))

!(add-reduct &kb (: treat-cold 
  (-> (Inheritance $x common_cold)
      (Evaluation recommend_rest_and_hydration $x))))

!(add-reduct &kb (: treat-common-cold
  (-> (Inheritance $x common_cold)
      (Evaluation recommend_warm_fluids_rest_steam $x))))

!(add-reduct &kb (: treat-asthma 
  (-> (Inheritance $x asthma)
      (Evaluation recommend_inhaler_and_breathing_exercises $x))))

!(add-reduct &kb (: treat-diabetes
  (-> (Inheritance $x diabetes)
      (Evaluation recommend_diet_exercise_monitoring $x))))

!(add-reduct &kb (: treat-hypertension
  (-> (Inheritance $x hypertension)
      (Evaluation recommend_dash_diet_stress_reduction $x))))

!(add-reduct &kb (: treat-acid-reflux
  (-> (Inheritance $x acid_reflux)
      (Evaluation recommend_dietary_changes_elevation $x))))

!(add-reduct &kb (: treat-insomnia
  (-> (Inheritance $x insomnia)
      (Evaluation recommend_sleep_hygiene_routine $x))))

!(add-reduct &kb (: treat-depression
  (-> (Inheritance $x depression)
      (Evaluation recommend_exercise_therapy_sunlight $x))))

!(add-reduct &kb (: treat-anxiety
  (-> (Inheritance $x anxiety)
      (Evaluation recommend_breathing_mindfulness_exercise $x))))

!(add-reduct &kb (: treat-arthritis
  (-> (Inheritance $x arthritis)
      (Evaluation recommend_gentle_exercise_heat_therapy $x))))

!(add-reduct &kb (: treat-ibs
  (-> (Inheritance $x ibs)
      (Evaluation recommend_diet_modification_stress_management $x))))

!(add-reduct &kb (: treat-migraine
  (-> (Inheritance $x migraine)
      (Evaluation recommend_trigger_avoidance_cold_compress $x))))

!(add-reduct &kb (: treat-seasonal-allergies
  (-> (Inheritance $x seasonal_allergies)
      (Evaluation recommend_avoidance_nasal_rinse $x))))

!(add-reduct &kb (: treat-constipation
  (-> (Inheritance $x constipation)
      (Evaluation recommend_fiber_hydration_exercise $x))))

!(add-reduct &kb (: treat-eczema
  (-> (Inheritance $x eczema)
      (Evaluation recommend_moisturize_avoid_triggers $x))))

!(add-reduct &kb (: treat-back-pain
  (-> (Inheritance $x back_pain)
      (Evaluation recommend_stretching_posture_heat $x))))

; Add patient symptoms as facts
;!(add-atom &kb (: FACT1 (Evaluation has_shortness_of_breath current_patient)))
;!(add-atom &kb (: FACT2 (Evaluation has_wheezing current_patient)))
;!(add-atom &kb (: FACT3 (Evaluation has_cough current_patient)))
;!(add-atom &kb (: FACT4 (Evaluation has_chest_tightness current_patient)))

; Run forward chaining from initial facts
;!(fcc &kb (fromNumber 4) (: FACT1 (Evaluation has_shortness_of_breath current_patient))) 