PRAGMA foreign_keys = ON;

CREATE TABLE samples (
  gsm TEXT PRIMARY KEY,
  global_index INTEGER NOT NULL UNIQUE,
  species TEXT NOT NULL,
  species_id INTEGER NOT NULL,
  shard_idx INTEGER,
  shard_file TEXT,
  row_in_shard INTEGER,
  CHECK (global_index >= 0)
);

CREATE TABLE sample_studies (
  gsm TEXT NOT NULL REFERENCES samples(gsm),
  gse TEXT NOT NULL,
  PRIMARY KEY (gsm, gse)
);

CREATE TABLE archs4_metadata (
  gsm TEXT PRIMARY KEY REFERENCES samples(gsm),
  geo_accession TEXT, series_id TEXT, organism_ch1 TEXT, taxid_ch1 TEXT,
  platform_id TEXT, title TEXT, source_name_ch1 TEXT, characteristics_ch1 TEXT,
  molecule_ch1 TEXT, library_strategy TEXT, library_source TEXT,
  library_selection TEXT, instrument_model TEXT, channel_count TEXT, type TEXT,
  sample TEXT, extract_protocol_ch1 TEXT, data_processing TEXT, relation TEXT,
  singlecellprobability REAL, submission_date TEXT, last_update_date TEXT,
  status TEXT, contact_name TEXT, contact_institute TEXT, contact_address TEXT,
  contact_city TEXT, contact_zip TEXT, contact_country TEXT,
  archs4_file TEXT NOT NULL, archs4_version TEXT NOT NULL
);

CREATE TABLE geo_downloads (
  gse TEXT PRIMARY KEY, status TEXT NOT NULL, http_status INTEGER, bytes INTEGER,
  sha256 TEXT, url TEXT, error TEXT, retrieved_at TEXT
);

CREATE TABLE studies (
  gse TEXT PRIMARY KEY, title TEXT, summary TEXT, overall_design TEXT,
  pubmed_ids TEXT, study_type TEXT, submission_date TEXT, last_update_date TEXT,
  organisms TEXT, taxids TEXT, platforms TEXT, bioproject_relations TEXT,
  sra_relations TEXT, other_relations TEXT, status TEXT, source_url TEXT,
  retrieved_at TEXT, source_sha256 TEXT
);

CREATE TABLE geo_sample_extras (
  gsm TEXT PRIMARY KEY REFERENCES samples(gsm),
  treatment_protocol TEXT, growth_protocol TEXT, description TEXT,
  geo_last_update_date TEXT, source_gse TEXT, source_url TEXT, retrieved_at TEXT
);

CREATE TABLE geometa (
  geometa_row_id INTEGER PRIMARY KEY,
  gsm TEXT NOT NULL, gse TEXT, year TEXT, contact_country TEXT,
  instrument_model TEXT, rna_library TEXT, exp_setting TEXT, gse_pert TEXT,
  disease TEXT, organ TEXT, gender TEXT, age_group TEXT,
  release_file TEXT NOT NULL, release_revision TEXT
);

CREATE TABLE gemma_annotations (
  gemma_annotation_id INTEGER PRIMARY KEY,
  gsm TEXT NOT NULL, gse TEXT, annotation_scope TEXT NOT NULL,
  category TEXT, category_uri TEXT, raw_value TEXT, value_uri TEXT,
  factor_id TEXT, factor_type TEXT, factor_value_ontology_id TEXT,
  gemma_sample_id TEXT, source_file TEXT NOT NULL
);

CREATE TABLE canonical_terms (
  canonical_id TEXT PRIMARY KEY, field TEXT NOT NULL, canonical_label TEXT NOT NULL,
  source_vocabulary TEXT NOT NULL, ontology_id TEXT, ontology_label TEXT,
  ontology_name TEXT, vocabulary_version TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1,
  UNIQUE(field, canonical_label, source_vocabulary, vocabulary_version)
);

CREATE TABLE canonical_mappings (
  mapping_id INTEGER PRIMARY KEY, source TEXT NOT NULL, field TEXT NOT NULL,
  raw_value TEXT NOT NULL, canonical_id TEXT REFERENCES canonical_terms(canonical_id),
  mapping_method TEXT NOT NULL, mapping_status TEXT NOT NULL,
  ontology_id TEXT, notes TEXT, UNIQUE(source, field, raw_value)
);

CREATE TABLE annotation_runs (
  annotation_run_id TEXT PRIMARY KEY, tool TEXT NOT NULL, model_name TEXT,
  model_version TEXT, model_commit TEXT, question_schema_version TEXT,
  canonical_vocabulary_version TEXT, runtime_environment TEXT, code_commit TEXT,
  run_date TEXT NOT NULL, notes TEXT
);

CREATE TABLE metadata_assertions (
  assertion_id INTEGER PRIMARY KEY,
  gsm TEXT NOT NULL REFERENCES samples(gsm), field TEXT NOT NULL,
  raw_value TEXT, canonical_value TEXT, canonical_id TEXT REFERENCES canonical_terms(canonical_id),
  source TEXT NOT NULL, annotation_method TEXT NOT NULL,
  model_name TEXT, model_version TEXT, model_commit TEXT, model_score REAL,
  evidence_text TEXT, evidence_source TEXT, ontology_id TEXT, ontology_name TEXT,
  annotation_run_id TEXT REFERENCES annotation_runs(annotation_run_id),
  schema_version TEXT NOT NULL, assertion_timestamp TEXT, status TEXT NOT NULL DEFAULT 'active'
);

CREATE TABLE canonical_metadata (
  gsm TEXT PRIMARY KEY REFERENCES samples(gsm),
  organ TEXT, disease TEXT, gender TEXT, age_group TEXT,
  experimental_setting TEXT, perturbation TEXT, cell_type TEXT, strain TEXT,
  genotype TEXT, treatment TEXT, dose TEXT, duration_development TEXT,
  organ_source TEXT, disease_source TEXT, gender_source TEXT, age_group_source TEXT,
  experimental_setting_source TEXT, perturbation_source TEXT,
  cell_type_source TEXT, strain_source TEXT, genotype_source TEXT,
  treatment_source TEXT, dose_source TEXT, duration_development_source TEXT,
  has_conflict INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE metadata_conflicts (
  conflict_id INTEGER PRIMARY KEY, gsm TEXT NOT NULL REFERENCES samples(gsm),
  field TEXT NOT NULL, source_a TEXT NOT NULL, value_a TEXT,
  source_b TEXT NOT NULL, value_b TEXT, status TEXT NOT NULL DEFAULT 'unresolved',
  notes TEXT
);

CREATE INDEX idx_sample_studies_gse ON sample_studies(gse);
CREATE INDEX idx_geometa_gsm ON geometa(gsm);
CREATE INDEX idx_gemma_gsm_field ON gemma_annotations(gsm, category);
CREATE INDEX idx_assertions_gsm_field ON metadata_assertions(gsm, field);
CREATE INDEX idx_assertions_source ON metadata_assertions(source);
CREATE INDEX idx_canonical_organ ON canonical_metadata(organ);
CREATE INDEX idx_canonical_species_lookup ON samples(species, gsm);
CREATE INDEX idx_conflicts_status ON metadata_conflicts(status);
