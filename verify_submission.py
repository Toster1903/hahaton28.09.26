"""Validate every output row against the archive and sample submission IDs."""
import argparse,csv,hashlib,io,json,zipfile
from pathlib import Path
from recommend import read_archive,validate_submission
root=Path(__file__).parent
p=argparse.ArgumentParser();p.add_argument('file',nargs='?',type=Path,default=root/'submission.csv');a=p.parse_args()
_,_,test,modules,_=read_archive(root/'data/selection_formula.zip')
with a.file.open(newline='',encoding='utf-8') as f:
 reader=csv.DictReader(f)
 if reader.fieldnames!=['id','recommendations']:raise ValueError('Wrong header')
 rows=list(reader)
if len(rows)!=len(test) or [r['id'] for r in rows]!=[r['id'] for r in test]:raise ValueError('Wrong IDs/order/row count')
with zipfile.ZipFile(root/'data/selection_formula.zip') as z:
 sample=list(csv.DictReader(io.StringIO(z.read('MCU/sample_submission.csv').decode())))
if [r['id'] for r in rows]!=[r['id'] for r in sample]:raise ValueError('IDs differ from sample submission')
validate_submission(test,{r['id']:r['recommendations'].split() for r in rows},{m['module_id'] for m in modules})
result={'rows':len(rows),'all_rows_valid':True,'sha256':hashlib.sha256(a.file.read_bytes()).hexdigest()}
print(json.dumps(result,indent=2))
if a.file.resolve()==(root/'submission.csv').resolve():(root/'audit/submission_checks.json').write_text(json.dumps(result,indent=2))
