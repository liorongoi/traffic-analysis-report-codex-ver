import argparse
import json
from pathlib import Path
from .domain import capture_time, named_zone
from .store import Library


def main():
    parser = argparse.ArgumentParser(description='Independent offline traffic analysis')
    sub = parser.add_subparsers(dest='command', required=True)
    stamp = sub.add_parser('timestamp')
    stamp.add_argument('filename')
    stamp.add_argument('--zone', default='UTC')
    stamp.add_argument('--preset', choices=['camera','fintraffic'],default='camera')
    run = sub.add_parser('serve')
    run.add_argument('--workspace',type=Path,default=Path.home()/'TrafficAnalysisWorkspace')
    run.add_argument('--port',type=int,default=8765)
    imp = sub.add_parser('import-fintraffic')
    imp.add_argument('source',type=Path)
    imp.add_argument('--workspace',type=Path,required=True)
    args = parser.parse_args()
    if args.command == 'timestamp':
        utc, precision = capture_time(args.filename,args.preset)
        print(json.dumps({'utc':utc.isoformat(),'precision':precision,'local':utc.astimezone(named_zone(args.zone)).isoformat()}))
    elif args.command == 'serve':
        from .server import serve
        serve(args.workspace,Path(__file__).resolve().parents[2]/'web',args.port)
    else:
        library=Library(args.workspace)
        mission=library.create('Fintraffic sample review','Finland','Europe/Helsinki')
        print(json.dumps({'mission':mission,**library.import_folder(mission,args.source,'fintraffic')}))

if __name__ == '__main__':
    main()
