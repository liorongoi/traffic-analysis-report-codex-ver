"""Run two detector packs on one frozen held-out COCO test split and compare them."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from road_traffic.benchmark import compare_detection_reports, evaluate_files, predict_coco_zip


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_name('.'+path.name+'.tmp')
    try:
        temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_prediction_bundle(path, bundle):
    bundle['metadata']['storage_bytes']['prediction_json']=0
    encoded=''
    for _ in range(5):
        encoded=json.dumps(bundle,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
        size=len(encoded.encode('utf-8'))
        if bundle['metadata']['storage_bytes']['prediction_json']==size:
            break
        bundle['metadata']['storage_bytes']['prediction_json']=size
    _write_json(path,bundle)


def compare_model_packs(dataset_zip, baseline_pack, candidate_pack, output_dir, *,
        model_kind='vehicle', baseline_threshold=None, candidate_threshold=None,
        tiled=True, operating_iou=.5, image_metadata=None, hardware_profile=None):
    """Evaluate both packs on the same held-out test ZIP and preserve all reports."""
    dataset_zip=Path(dataset_zip);baseline_pack=Path(baseline_pack)
    candidate_pack=Path(candidate_pack);output_dir=Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'Output directory must be empty: {output_dir}')
    output_dir.mkdir(parents=True,exist_ok=True)
    run_metadata={'hardware_profile':hardware_profile} if hardware_profile else None
    report_paths=[]
    for label,model_pack,threshold in (
            ('baseline',baseline_pack,baseline_threshold),
            ('candidate',candidate_pack,candidate_threshold)):
        print(f'Predicting held-out test images with {label} model: {model_pack}')
        bundle=predict_coco_zip(dataset_zip,model_pack,split='test',model_kind=model_kind,
            threshold=threshold,tiled=tiled,image_metadata=image_metadata,strata=run_metadata)
        prediction_path=output_dir/f'{label}-predictions.json'
        report_path=output_dir/f'{label}-report.json'
        _write_prediction_bundle(prediction_path,bundle)
        evaluate_files(dataset_zip,prediction_path,report_path,split='test',operating_iou=operating_iou)
        report_paths.append(report_path)
    baseline=json.loads(report_paths[0].read_text(encoding='utf-8'))
    candidate=json.loads(report_paths[1].read_text(encoding='utf-8'))
    comparison=compare_detection_reports(baseline,candidate)
    _write_json(output_dir/'comparison.json',comparison)
    return comparison


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-zip',required=True,type=Path,
        help='Frozen RoadLoom COCO ZIP with complete independent labels in test/')
    parser.add_argument('--baseline-model-pack',required=True,type=Path)
    parser.add_argument('--candidate-model-pack',required=True,type=Path)
    parser.add_argument('--model-kind',choices=('vehicle','plate'),default='vehicle')
    parser.add_argument('--baseline-threshold',type=float,
        help='Baseline threshold; defaults to 0.18 for vehicles or 0.30 for plate regions')
    parser.add_argument('--candidate-threshold',type=float,
        help='Candidate threshold; tune on validation data before the one-time test evaluation')
    parser.add_argument('--no-tiles',action='store_true',help='Disable vehicle tiling for both models')
    parser.add_argument('--operating-iou',type=float,default=.5)
    parser.add_argument('--image-metadata',type=Path,
        help='Optional JSON map from COCO image IDs to country/condition/view labels')
    parser.add_argument('--hardware-profile',help='Run-level hardware label for stratified reporting')
    parser.add_argument('--output-dir',required=True,type=Path,
        help='New or empty directory for predictions, evaluation reports and comparison JSON')
    args=parser.parse_args()
    if args.image_metadata:
        image_metadata=json.loads(args.image_metadata.read_text(encoding='utf-8'))
    else:
        image_metadata=None
    result=compare_model_packs(args.dataset_zip,args.baseline_model_pack,
        args.candidate_model_pack,args.output_dir,model_kind=args.model_kind,
        baseline_threshold=args.baseline_threshold,candidate_threshold=args.candidate_threshold,
        tiled=not args.no_tiles,operating_iou=args.operating_iou,
        image_metadata=image_metadata,hardware_profile=args.hardware_profile)
    overall=result['changes']['overall']['micro']
    print(f"Compared {result['images']} held-out test images; report: {args.output_dir/'comparison.json'}")
    print('Candidate-minus-baseline P/R/F1: {}/{}/{}'.format(
        overall['precision']['delta'],overall['recall']['delta'],overall['f1']['delta']))
    print(f"Improvements: {len(result['improvements'])}; regressions: {len(result['regressions'])}")
    print('No acceptance or model-promotion decision was made; inspect all reports and strata.')


if __name__=='__main__':
    main()
