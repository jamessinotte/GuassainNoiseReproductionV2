"""Run checks and write an honest machine-readable validation record."""
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from repro.io import environment,write_json


def main():
    output=ROOT/"results/validation"
    output.mkdir(parents=True,exist_ok=True)
    checks={}
    for name,command in [("tests",[sys.executable,"-m","pytest","-q"]),
                         ("dependencies",[sys.executable,"-m","pip","check"]),
                         ("compile",[sys.executable,"-m","compileall","-q","repro","scripts","tests"])]:
        result=subprocess.run(command,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (output/(name+".log")).write_text(result.stdout)
        checks[name]={"command":command,"returncode":result.returncode,"output":result.stdout}
        print(name,result.returncode,result.stdout[-1200:],flush=True)
    freeze=subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True)
    (output/"requirements-resolved.txt").write_text(freeze)
    write_json(output/"record.json",{"executed_utc":datetime.now(timezone.utc).isoformat(),"checks":checks,
                                     "environment":environment(),"gpu_inference_executed":False,
                                     "full_research_sweeps_executed":False})
    return int(any(c["returncode"] for c in checks.values()))


if __name__=="__main__":
    raise SystemExit(main())
