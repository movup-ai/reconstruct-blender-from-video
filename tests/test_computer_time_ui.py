"""Exercise the real table renderer with cumulative timing responses."""
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT=Path(__file__).resolve().parents[1]


class ComputerTimeUiTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'),'Node is needed for JavaScript rendering checks')
    def test_table_uses_cumulative_time(self):
        script=r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert'),root=process.argv[1];
const context=vm.createContext({localStorage:{removeItem(){}}});
vm.runInContext(fs.readFileSync(root+'/web/common.js','utf8'),context);
const index=fs.readFileSync(root+'/web/index.html','utf8');
const table=index.slice(index.indexOf('function renderTable('),index.indexOf('function progressSummary('));
vm.runInContext(`function clipActions(){return {action:''};} function statusCell(){return '';}`+table,context);
const runs={done:{machine:180}};
const cumulative={...runs,machine:780};
assert.equal(vm.runInContext('computerSeconds(null)',context),null);
context.timing=runs;assert.equal(vm.runInContext('computerSeconds(timing)',context),180);
context.timing=cumulative;assert.equal(vm.runInContext('computerSeconds(timing)',context),780);
vm.runInContext(`const list={};renderTable(list,[{id:'a',name:'clip',status:{},took:timing}],{});
 if(!list.innerHTML.includes('>13 min</td>'))throw Error('Table displayed the last action instead of total');`,context);
context.timing={...cumulative,machine:880,open:{machine:100}};
assert.equal(vm.runInContext('computerSeconds(timing)',context),880);
context.timing={machine:0};assert.equal(vm.runInContext('computerSeconds(timing)',context),0);
'''
        result=subprocess.run(['node','-e',script,str(ROOT)],capture_output=True,text=True,encoding='utf-8')
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
