#!/usr/bin/env python3
"""Small end-to-end validation of both protocols before launching inference."""
import json
import time
import numpy as np
from common import CONFIG, RESULTS
from fresh_run import check_inputs, deterministic, cache_model
from evaluation import MODELS, MODEL_SPACE, condition_masks, apply_mask, score_prediction
import model_adapters as adapters

if __name__=='__main__':
    samples,ours,bulk,og,bg,shared=check_inputs();ids=samples.sample_id.astype(str).tolist()
    original_ours,original_bulk=adapters.load_ours,adapters.load_bulkformer
    rows=[]
    for name in MODELS:
        adapters.load_ours,adapters.load_bulkformer=original_ours,original_bulk
        model,device=cache_model(name)
        for mode in ['shared_vocab','native_vocab']:
            for ratio in [.15,1.0]:
                deterministic(0)
                mask,_=condition_masks(mode,name,ids,ratio,0,og,bg,shared)
                truth=(ours if MODEL_SPACE[name]=='ours' else bulk)[:2];mask=mask[:2]
                x=apply_mask(truth,mask)
                pred=adapters.predict(name,x,float((x==CONFIG['mask_token']).mean()),device,2)
                assert pred.shape==truth.shape and np.isfinite(pred).all()
                scores=score_prediction(ids[:2],truth,pred,mask)
                assert np.isfinite(scores.mse).all()
                if mode=='shared_vocab' and ratio==.15:
                    deterministic(0)
                    again=adapters.predict(name,x,float((x==CONFIG['mask_token']).mean()),device,2)
                    repeat_scores=score_prediction(ids[:2],truth,again,mask)
                    delta={metric:float(np.nanmax(np.abs(scores[metric]-repeat_scores[metric]))) for metric in ['pearson','spearman','mse']}
                    difference={'model':name,'max_absolute_prediction_difference':float(np.max(np.abs(pred-again))), 'max_metric_difference':delta}
                    print(json.dumps(difference),flush=True)
                    (RESULTS/f'repeatability_{name}.json').write_text(json.dumps(difference,indent=2)+'\n')
                    assert max(delta.values()) < .001, f'Material repeatability difference: {delta}'
                rows.append({'model':name,'benchmark':mode,'mask_ratio':ratio,'finite_output':True,
                             'defined_pearson':int(scores.pearson.notna().sum()),'samples':2})
        del model
    (RESULTS/'protocol_validation.json').write_text(json.dumps({'status':'passed','conditions':rows},indent=2)+'\n')
    print('Both protocols passed small-cohort validation for all models.',flush=True)
