"""Compare two RoadLoom detector reports on one frozen held-out test split."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from road_traffic.benchmark import compare_detection_reports


def _read_report(path):
    try:
        report=json.loads(path.read_text(encoding='utf-8'))
    except (OSError,json.JSONDecodeError) as exc:
        raise ValueError(f'Unable to read benchmark report {path}: {exc}') from exc
    if not isinstance(report,dict):
        raise ValueError(f'Benchmark report must be a JSON object: {path}')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-report',required=True,type=Path,
        help='Detection benchmark report for the currently selected model')
    parser.add_argument('--candidate-report',required=True,type=Path,
        help='Detection benchmark report for the candidate model')
    parser.add_argument('--output',required=True,type=Path,help='Output JSON comparison report')
    args=parser.parse_args()
    report=compare_detection_reports(_read_report(args.baseline_report),_read_report(args.candidate_report))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_name('.'+args.output.name+'.tmp')
    try:
        temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    overall=report['changes']['overall']['micro']
    print(f"Compared {report['images']} held-out test images; report: {args.output}")
    print('Candidate-minus-baseline P/R/F1: {}/{}/{}'.format(
        overall['precision']['delta'],overall['recall']['delta'],overall['f1']['delta']))
    print(f"Improvements: {len(report['improvements'])}; regressions: {len(report['regressions'])}")
    print('No model-promotion decision was made; review the full report and strata.')


if __name__=='__main__':
    main()
