import threading

from fastapi.testclient import TestClient

from tools import plan_runner_gui as gui


def test_run_resolves_named_variant_without_changing_active_config(monkeypatch, tmp_path):
    seen = {}
    done = threading.Event()
    def run(plan, **kwargs):
        seen.update(kwargs)
        done.set()
        return dict(plan=plan, score=1, arrived=True, run_dir=None)
    monkeypatch.setattr(gui, 'run_single', run)
    monkeypatch.setattr(gui, 'DEFAULT_LOG_DIR', tmp_path)
    response = TestClient(gui.app).post('/api/run', json={
        'plan': 'real_calibrated', 'variant': 'x_forward_apf', 'max_time': 1, 'debris_count': 50,
    })
    assert response.status_code == 200
    assert done.wait(2)
    assert seen['config_overrides']['burn'] == 'target_x'
    assert seen['config_overrides']['apf'] is True
    assert seen['debris_count'] == 50


def test_run_rejects_unknown_variant():
    response = TestClient(gui.app).post('/api/run', json={
        'plan': 'real_calibrated', 'variant': 'missing_variant', 'max_time': 1,
    })
    assert response.status_code == 422


def test_variant_get_and_invalid_density_combination():
    client = TestClient(gui.app)
    response = client.get('/api/plans/real_calibrated/variants/x_forward_apf')
    assert response.status_code == 200
    assert response.json()['config']['burn'] == 'target_x'
    assert client.get('/api/plans/real_calibrated/variants/absent').status_code == 422
    assert client.post('/api/run', json={'plan': 'astar', 'scenario': 'point_mass', 'debris_count': 50}).status_code == 422


def test_run_variant_dropdown_lists_presets_and_resets_on_plan_change():
    import shutil
    import subprocess
    from pathlib import Path
    import pytest
    if not shutil.which('node'):
        pytest.skip('node is needed for the dropdown unit test')
    text = (Path(__file__).resolve().parents[1] / 'tools/gui_static.html').read_text(encoding='utf-8')
    start = text.index('function renderVariantInfo(p) {')
    end = text.index('\nfunction updateParamModBadge', start)
    script = '''
const assert = require('node:assert/strict');
let runVariantRequest=0,runVariantLoading=false,runVariantError='';
class Option { constructor(text,value) { this.textContent=text; this.value=value; } }
class Select { constructor() { this.options=[];this.dataset={};this.value=''; } set innerHTML(v) {this.options=[];} appendChild(o) {this.options.push(o);} add(o) {this.options.push(o);} replaceChildren(...o) {this.options=o;} }
const elements={variantSelect:new Select(),runVariantSelect:new Select(),symlinkBadge:{}};
const document={getElementById:k=>elements[k],createElement:()=>new Option('','')};
''' + text[start:end] + '''
renderVariantInfo({name:'real_calibrated',variant_info:{variants:[{name:'default',active:true},{name:'x_forward'},{name:'x_forward_apf'}]}});
assert.deepEqual(elements.runVariantSelect.options.map(o=>o.value),['','default','x_forward','x_forward_apf']);
elements.runVariantSelect.value='x_forward_apf';
renderVariantInfo({name:'real_calibrated',variant_info:{variants:[{name:'default'},{name:'x_forward_apf'}]}});
assert.equal(elements.runVariantSelect.value,'x_forward_apf');
renderVariantInfo({name:'astar',variant_info:{variants:[{name:'default'}]}});
assert.equal(elements.runVariantSelect.value,'');
'''
    result = subprocess.run(['node', '-e', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
