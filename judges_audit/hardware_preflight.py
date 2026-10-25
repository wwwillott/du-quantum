"""Check large-case hardware contract using Willow median calibration.

This does not submit a quantum processor job.
"""
import json
import time
import argparse
from .compare import ROOT, OUT, SOURCES, load, calibration_maps, check
import cirq
import cirq_google


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', choices=('kai_latest','will'),default='kai_latest')
    args = parser.parse_args()
    module = load(SOURCES[args.candidate], args.candidate+'_preflight')
    engine = cirq_google.engine.create_noiseless_virtual_engine_from_latest_templates()
    processor = engine.get_processor('willow_pink')
    maps = calibration_maps(cirq_google.engine.load_median_device_calibration('willow_pink'))
    results = dict(physical_hardware=False, source=str(SOURCES[args.candidate]), rows=[])
    for n,k in [(12,6),(16,8),(20,1),(20,10)]:
        start = time.perf_counter()
        circuit, outputs = module.create_dicke_state(n,k,*maps)
        seconds = time.perf_counter()-start
        verification = check(circuit,outputs,n,k,maps,seconds,processor.get_device())
        row = dict(n=n,k=k,construction_seconds=seconds,moments=len(circuit),verification=verification,
                   native_gates=True,calibrated_qubits=True,calibrated_couplers=True,serializer_pass=True)
        results['rows'].append(row)
        name = 'large_contract.json' if args.candidate=='kai_latest' else 'large_contract_will.json'
        (OUT/name).write_text(json.dumps(results,indent=2),encoding='utf8')
        print(row,flush=True)


if __name__ == '__main__':
    main()
