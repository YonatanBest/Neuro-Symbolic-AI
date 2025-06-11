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

;; Treatment Rules

;; Treatment Rules for Asthma
!(add-reduct &kb (: asthma-treatment-rule 
    (-> (Inheritance $x asthma)
        (Treatment $x (use_inhaler avoid_triggers practice_breathing_exercises)))))

!(add-reduct &kb (: asthma-medication-rule 
    (-> (Inheritance $x asthma)
        (Medication $x (quick_relief_inhaler long_term_control_inhaler oral_medications)))))

!(add-reduct &kb (: asthma-lifestyle-rule 
    (-> (Inheritance $x asthma)
        (Lifestyle $x (avoid_smoke avoid_allergens exercise_regularly maintain_healthy_weight)))))

!(add-reduct &kb (: asthma-monitoring-rule 
    (-> (Inheritance $x asthma)
        (Monitoring $x (track_symptoms use_peak_flow_meter regular_checkups)))))

;; Treatment Rules for Flu
!(add-reduct &kb (: flu-treatment-rule 
    (-> (Inheritance $x flu)
        (Treatment $x (rest hydration otc_pain_relievers antiviral_medications)))))

!(add-reduct &kb (: flu-medication-rule 
    (-> (Inheritance $x flu)
        (Medication $x (acetaminophen ibuprofen oseltamivir zanamivir)))))

!(add-reduct &kb (: flu-lifestyle-rule 
    (-> (Inheritance $x flu)
        (Lifestyle $x (stay_home rest sleep_well avoid_contact)))))

!(add-reduct &kb (: flu-monitoring-rule 
    (-> (Inheritance $x flu)
        (Monitoring $x (check_temperature watch_symptoms seek_help_if_worsening)))))

;; Treatment Rules for COVID-19
!(add-reduct &kb (: covid-treatment-rule 
    (-> (Inheritance $x covid_19)
        (Treatment $x (isolation supportive_care oxygen_therapy antiviral_therapy)))))

!(add-reduct &kb (: covid-medication-rule 
    (-> (Inheritance $x covid_19)
        (Medication $x (antipyretics pain_relievers prescribed_antivirals monoclonal_antibodies)))))

!(add-reduct &kb (: covid-lifestyle-rule 
    (-> (Inheritance $x covid_19)
        (Lifestyle $x (strict_isolation rest hydration monitor_oxygen_levels)))))

!(add-reduct &kb (: covid-monitoring-rule 
    (-> (Inheritance $x covid_19)
        (Monitoring $x (pulse_oximeter temperature_checks symptom_diary emergency_signs)))))

;; Treatment Rules for Common Cold
!(add-reduct &kb (: cold-treatment-rule 
    (-> (Inheritance $x common_cold)
        (Treatment $x (rest hydration otc_medications steam_inhalation)))))

!(add-reduct &kb (: cold-medication-rule 
    (-> (Inheritance $x common_cold)
        (Medication $x (decongestants antihistamines cough_suppressants zinc_supplements)))))

!(add-reduct &kb (: cold-lifestyle-rule 
    (-> (Inheritance $x common_cold)
        (Lifestyle $x (rest sleep gargle_saltwater humidifier)))))

!(add-reduct &kb (: cold-monitoring-rule 
    (-> (Inheritance $x common_cold)
        (Monitoring $x (track_duration watch_fever check_congestion)))))

;; Treatment Rules for Diabetes
!(add-reduct &kb (: diabetes-treatment-rule 
    (-> (Inheritance $x diabetes)
        (Treatment $x (blood_sugar_monitoring insulin_therapy diet_management exercise_program)))))

!(add-reduct &kb (: diabetes-medication-rule 
    (-> (Inheritance $x diabetes)
        (Medication $x (insulin metformin sulfonylureas glp1_agonists)))))

!(add-reduct &kb (: diabetes-lifestyle-rule 
    (-> (Inheritance $x diabetes)
        (Lifestyle $x (healthy_diet regular_exercise weight_management stress_reduction)))))

!(add-reduct &kb (: diabetes-monitoring-rule 
    (-> (Inheritance $x diabetes)
        (Monitoring $x (blood_glucose_checks a1c_tests foot_checks eye_exams)))))

;; Treatment Rules for Hypertension
!(add-reduct &kb (: hypertension-treatment-rule 
    (-> (Inheritance $x hypertension)
        (Treatment $x (blood_pressure_monitoring medication_management dietary_changes stress_reduction)))))

!(add-reduct &kb (: hypertension-medication-rule 
    (-> (Inheritance $x hypertension)
        (Medication $x (ace_inhibitors beta_blockers diuretics calcium_channel_blockers)))))

!(add-reduct &kb (: hypertension-lifestyle-rule 
    (-> (Inheritance $x hypertension)
        (Lifestyle $x (reduce_sodium exercise_regularly limit_alcohol maintain_healthy_weight)))))

!(add-reduct &kb (: hypertension-monitoring-rule 
    (-> (Inheritance $x hypertension)
        (Monitoring $x (daily_bp_readings diet_tracking exercise_log medication_compliance)))))

;; Treatment Rules for Acid Reflux
!(add-reduct &kb (: acid-reflux-treatment-rule 
    (-> (Inheritance $x acid_reflux)
        (Treatment $x (dietary_modifications elevation_while_sleeping antacids lifestyle_changes)))))

!(add-reduct &kb (: acid-reflux-medication-rule 
    (-> (Inheritance $x acid_reflux)
        (Medication $x (proton_pump_inhibitors h2_blockers antacids alginates)))))

!(add-reduct &kb (: acid-reflux-lifestyle-rule 
    (-> (Inheritance $x acid_reflux)
        (Lifestyle $x (avoid_trigger_foods eat_smaller_meals no_late_eating elevate_head_of_bed)))))

!(add-reduct &kb (: acid-reflux-monitoring-rule 
    (-> (Inheritance $x acid_reflux)
        (Monitoring $x (track_trigger_foods symptom_diary weight_management)))))

;; Treatment Rules for Insomnia
!(add-reduct &kb (: insomnia-treatment-rule 
    (-> (Inheritance $x insomnia)
        (Treatment $x (sleep_hygiene cognitive_behavioral_therapy relaxation_techniques medication)))))

!(add-reduct &kb (: insomnia-medication-rule 
    (-> (Inheritance $x insomnia)
        (Medication $x (sleep_medications melatonin antihistamines prescription_sleep_aids)))))

!(add-reduct &kb (: insomnia-lifestyle-rule 
    (-> (Inheritance $x insomnia)
        (Lifestyle $x (regular_sleep_schedule avoid_screens limit_caffeine exercise_early)))))

!(add-reduct &kb (: insomnia-monitoring-rule 
    (-> (Inheritance $x insomnia)
        (Monitoring $x (sleep_diary sleep_quality_tracking daytime_symptoms)))))

;; Treatment Rules for Depression
!(add-reduct &kb (: depression-treatment-rule 
    (-> (Inheritance $x depression)
        (Treatment $x (psychotherapy medication_management lifestyle_changes support_groups)))))

!(add-reduct &kb (: depression-medication-rule 
    (-> (Inheritance $x depression)
        (Medication $x (ssris snris ndris mood_stabilizers)))))

!(add-reduct &kb (: depression-lifestyle-rule 
    (-> (Inheritance $x depression)
        (Lifestyle $x (regular_exercise social_connection sunlight_exposure healthy_diet)))))

!(add-reduct &kb (: depression-monitoring-rule 
    (-> (Inheritance $x depression)
        (Monitoring $x (mood_tracking symptom_diary therapy_progress medication_effects)))))

;; Treatment Rules for Anxiety
!(add-reduct &kb (: anxiety-treatment-rule 
    (-> (Inheritance $x anxiety)
        (Treatment $x (psychotherapy medication relaxation_techniques mindfulness)))))

!(add-reduct &kb (: anxiety-medication-rule 
    (-> (Inheritance $x anxiety)
        (Medication $x (ssris benzodiazepines buspirone beta_blockers)))))

!(add-reduct &kb (: anxiety-lifestyle-rule 
    (-> (Inheritance $x anxiety)
        (Lifestyle $x (stress_management regular_exercise meditation healthy_sleep)))))

!(add-reduct &kb (: anxiety-monitoring-rule 
    (-> (Inheritance $x anxiety)
        (Monitoring $x (anxiety_levels trigger_tracking coping_strategies_effectiveness)))))

;; Treatment Rules for Arthritis
!(add-reduct &kb (: arthritis-treatment-rule 
    (-> (Inheritance $x arthritis)
        (Treatment $x (physical_therapy medication pain_management joint_protection)))))

!(add-reduct &kb (: arthritis-medication-rule 
    (-> (Inheritance $x arthritis)
        (Medication $x (nsaids dmards corticosteroids biologics)))))

!(add-reduct &kb (: arthritis-lifestyle-rule 
    (-> (Inheritance $x arthritis)
        (Lifestyle $x (gentle_exercise weight_management joint_care assistive_devices)))))

!(add-reduct &kb (: arthritis-monitoring-rule 
    (-> (Inheritance $x arthritis)
        (Monitoring $x (pain_levels joint_mobility medication_effectiveness)))))

;; Treatment Rules for IBS
!(add-reduct &kb (: ibs-treatment-rule 
    (-> (Inheritance $x ibs)
        (Treatment $x (dietary_management stress_reduction medication gut_health_support)))))

!(add-reduct &kb (: ibs-medication-rule 
    (-> (Inheritance $x ibs)
        (Medication $x (antispasmodics laxatives antidiarrheals probiotics)))))

!(add-reduct &kb (: ibs-lifestyle-rule 
    (-> (Inheritance $x ibs)
        (Lifestyle $x (fodmap_diet stress_management regular_exercise adequate_hydration)))))

!(add-reduct &kb (: ibs-monitoring-rule 
    (-> (Inheritance $x ibs)
        (Monitoring $x (symptom_diary food_diary stress_levels bowel_habits)))))

;; Treatment Rules for Migraine
!(add-reduct &kb (: migraine-treatment-rule 
    (-> (Inheritance $x migraine)
        (Treatment $x (acute_medication preventive_therapy trigger_avoidance stress_management)))))

!(add-reduct &kb (: migraine-medication-rule 
    (-> (Inheritance $x migraine)
        (Medication $x (triptans nsaids preventive_medications antiemetics)))))

!(add-reduct &kb (: migraine-lifestyle-rule 
    (-> (Inheritance $x migraine)
        (Lifestyle $x (regular_sleep stress_reduction trigger_avoidance hydration)))))

!(add-reduct &kb (: migraine-monitoring-rule 
    (-> (Inheritance $x migraine)
        (Monitoring $x (headache_diary trigger_tracking medication_effectiveness)))))

;; Treatment Rules for Seasonal Allergies
!(add-reduct &kb (: allergy-treatment-rule 
    (-> (Inheritance $x seasonal_allergies)
        (Treatment $x (medication allergen_avoidance nasal_irrigation immunotherapy)))))

!(add-reduct &kb (: allergy-medication-rule 
    (-> (Inheritance $x seasonal_allergies)
        (Medication $x (antihistamines decongestants nasal_steroids eye_drops)))))

!(add-reduct &kb (: allergy-lifestyle-rule 
    (-> (Inheritance $x seasonal_allergies)
        (Lifestyle $x (monitor_pollen_levels keep_windows_closed use_air_purifiers shower_after_outdoors)))))

!(add-reduct &kb (: allergy-monitoring-rule 
    (-> (Inheritance $x seasonal_allergies)
        (Monitoring $x (symptom_tracking pollen_forecasts medication_effectiveness)))))

;; Treatment Rules for Constipation
!(add-reduct &kb (: constipation-treatment-rule 
    (-> (Inheritance $x constipation)
        (Treatment $x (dietary_changes hydration medication exercise)))))

!(add-reduct &kb (: constipation-medication-rule 
    (-> (Inheritance $x constipation)
        (Medication $x (fiber_supplements stool_softeners osmotic_laxatives stimulant_laxatives)))))

!(add-reduct &kb (: constipation-lifestyle-rule 
    (-> (Inheritance $x constipation)
        (Lifestyle $x (high_fiber_diet regular_exercise adequate_hydration bathroom_routine)))))

!(add-reduct &kb (: constipation-monitoring-rule 
    (-> (Inheritance $x constipation)
        (Monitoring $x (bowel_movement_diary dietary_tracking hydration_levels)))))

;; Treatment Rules for Eczema
!(add-reduct &kb (: eczema-treatment-rule 
    (-> (Inheritance $x eczema)
        (Treatment $x (moisturizing trigger_avoidance medication skin_care)))))

!(add-reduct &kb (: eczema-medication-rule 
    (-> (Inheritance $x eczema)
        (Medication $x (topical_steroids immunosuppressants antihistamines moisturizers)))))

!(add-reduct &kb (: eczema-lifestyle-rule 
    (-> (Inheritance $x eczema)
        (Lifestyle $x (gentle_skin_care avoid_triggers humidity_control stress_management)))))

!(add-reduct &kb (: eczema-monitoring-rule 
    (-> (Inheritance $x eczema)
        (Monitoring $x (skin_condition_tracking trigger_diary treatment_effectiveness)))))

;; Treatment Rules for Back Pain
!(add-reduct &kb (: back-pain-treatment-rule 
    (-> (Inheritance $x back_pain)
        (Treatment $x (physical_therapy pain_management exercise posture_correction)))))

!(add-reduct &kb (: back-pain-medication-rule 
    (-> (Inheritance $x back_pain)
        (Medication $x (nsaids muscle_relaxants topical_treatments pain_relievers)))))

!(add-reduct &kb (: back-pain-lifestyle-rule 
    (-> (Inheritance $x back_pain)
        (Lifestyle $x (proper_posture ergonomic_adjustments core_strengthening weight_management)))))

!(add-reduct &kb (: back-pain-monitoring-rule 
    (-> (Inheritance $x back_pain)
        (Monitoring $x (pain_levels activity_impact treatment_effectiveness)))))

; Add patient symptoms as facts
;!(add-atom &kb (: FACT1 (Evaluation has_shortness_of_breath current_patient)))
;!(add-atom &kb (: FACT2 (Evaluation has_wheezing current_patient)))
;!(add-atom &kb (: FACT3 (Evaluation has_cough current_patient)))
;!(add-atom &kb (: FACT4 (Evaluation has_chest_tightness current_patient)))

; Run forward chaining from initial facts
;!(fcc &kb (fromNumber 4) (: FACT1 (Evaluation has_shortness_of_breath current_patient))) 