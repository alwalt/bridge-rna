-- 1. Retrieve canonical and original metadata for a GSM.
SELECT s.*, c.*, a.title, a.source_name_ch1, a.characteristics_ch1
FROM samples s
JOIN canonical_metadata c USING (gsm)
JOIN archs4_metadata a USING (gsm)
WHERE s.gsm = 'GSM1000562';

-- 2. Retrieve metadata for an embedding global_index.
SELECT s.global_index, s.gsm, c.*
FROM samples s JOIN canonical_metadata c USING (gsm)
WHERE s.global_index = 0;

-- 3. Retrieve all study context for a GSM.
SELECT ss.gsm, st.*
FROM sample_studies ss JOIN studies st USING (gse)
WHERE ss.gsm = 'GSM1000562';

-- 4. Filter samples by canonical organ.
SELECT s.global_index, s.gsm, c.organ
FROM samples s JOIN canonical_metadata c USING (gsm)
WHERE c.organ = 'Brain';

-- 5. Filter mouse liver samples.
SELECT s.global_index, s.gsm
FROM samples s JOIN canonical_metadata c USING (gsm)
WHERE s.species = 'mouse' AND c.organ = 'Liver';

-- 6. Retrieve all source assertions for a field.
SELECT * FROM metadata_assertions
WHERE gsm = 'GSM1000562' AND field = 'organ'
ORDER BY assertion_id;

-- 7. Identify model-generated labels.
SELECT * FROM metadata_assertions
WHERE annotation_method = 'model_generated';

-- 8. Show Laya model/version/score/evidence for a label.
SELECT m.gsm, m.field, m.canonical_value, m.model_name, m.model_version,
       m.model_commit, m.model_score, m.evidence_text, r.question_schema_version
FROM metadata_assertions m
JOIN annotation_runs r USING (annotation_run_id)
WHERE m.source = 'Laya';

-- 9. Find unresolved/conflicting metadata.
SELECT * FROM metadata_conflicts WHERE status = 'unresolved';
