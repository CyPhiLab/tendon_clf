# Drive-model study (docs/drive_model.md)

Scripts that compared the drive models A-E on the 20261103 runs. Not used at run time: the chosen
model lives in `spirob_zmq/drive.py`.

```bash
export SYSID_DATA=<unzipped drive_model_bundle/data>    # the recordings (not in the repo)
cd tools/drive_model_study
python harness.py '{"drive":"E"}' '["20261103_174014"]'          # replay one option, report slips and markers
python fit.py <tag> local '<start cfg>' '<space>' <rounds>          # coordinate search (see the docstring)
python evaluate.py options.json out/ && python table.py out/results.json A0,A,B,C,D,E
python plots.py out/results.json out/plots A0,C,E
```

- `drives.py`: the five variants (A current source + Coulomb tendon friction, B tendon stick-slip,
  C motor DOF + Karnopp, D dcmotor + LuGre on the robot tendon, E motor DOF + dcmotor LuGre) and the
  give (`give='rope'` per-tendon spring, `give='common'` compression carriage).
- `harness.py`: open-loop replay from the fitted start pose, marker rms per phase, slips from motor_state,
  spring-back on unloading, the fit objective.
- `options.json`: the evaluated configurations (fitted values).
