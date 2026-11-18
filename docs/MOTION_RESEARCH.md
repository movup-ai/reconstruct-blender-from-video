# Real human movement — reference values used by `pipeline/motion_prior.py`

These values are a **soft recommendation**. The detected joints (yellow dots) stay the main truth: the prior only
moves a joint in proportion to how unsure its dots are (`strength × (1 − dot confidence)`, strength 0.6).
Enabled per clip in `hints.json` with `"motion_prior": true`; per person `"motion_strength"`, `"motion_style": "kimono"`,
`"motion_prior": false`.

| Topic | Value used | Source |
| --- | --- | --- |
| Walking gait cycle | stance ~60 %, swing ~40 % | [Missouri PM&R, Normal Gait](https://www.medicine.missouri.edu/sites/default/files/pm-r/2020_Normal_Gait.pdf), [Wikipedia: Lower-limb walking pattern](https://en.wikipedia.org/wiki/Lower-limb_walking_pattern) |
| Hip in walking | ~30° flexion at heel contact, ~10–15° extension before toe-off | Wikipedia (above) |
| Knee in walking | near straight at heel contact, 10–15° early stance, ~60° peak in swing | Wikipedia; [PMC555760 table](https://pmc.ncbi.nlm.nih.gov/articles/PMC555760/table/T3) |
| Step length / speed | step 0.68–0.85 m, speed ~1.3–1.46 m/s | [Frontiers in Neurology 2021, table 3](https://pmc.ncbi.nlm.nih.gov/articles/PMC8217618/table/T3) |
| Cadence | ~90–113 steps/min depending on height; independent of age | [Rowe et al. 2011](https://strathprints.strath.ac.uk/35536/), [Samson et al. 2001](https://pubmed.ncbi.nlm.nih.gov/11292147/) |
| Arm swing | shoulder flexion-extension ~20–25° (≈24.6° amplitude), opposite to the same-side leg; elbow 15–30° | [Vandewalle et al. 2018](https://documentserver.uhasselt.be/handle/1942/26634), [Udochkina et al.](https://link.springer.com/article/10.1007/s11055-018-0647-8) |
| Arms in running | elbow bent (~80–90°), straight-ish in walking | [Straight arm walking, bent arm running](https://ouci.dntb.gov.ua/works/40zyKK59) |
| Arm swing role | damps trunk/pelvis rotation | [Arellano & Kram, J Exp Biol 2014](https://grants.hhp.uh.edu/arellano/docs/J%20Exp%20Biol%202014%20Arellano.pdf) |
| Centre of mass | vertical ~5 cm (≈2 % of height), lateral ~4–6 cm | [OUHSC gait kinematics](https://ouhsc.edu/bserdac/dthompso/web/gait/knmatics/gait.htm) |
| Running | hip ~45–50° at foot strike, ~55° swing; knee 25–40° stance, 90–125° peak swing; duty factor 25–40 % | [Biomechanics of Running](https://gyansanchay.csjmu.ac.in/wp-content/uploads/2022/02/Biomechanics-of-Running.pdf), [Patoz et al. 2020](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2020.625557/pdf) |
| Joint ranges | hip flex 120 / ext 20 / abd 45 / add 30; knee ~135–150; elbow ~140; shoulder flex 180 / ext 60 | [Physiopedia ROM normative values](https://www.physio-pedia.com/Range_of_Motion_Normative_Values), [CDC Joint ROM study](https://archive.cdc.gov/www_cdc_gov/ncbddd/jointrom/index.html) |
| Kimono gait | shorter steps, trunk rotation restrained | [Ota et al. 2017, PLoS ONE](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5480874/) |
| Kneeling / seiza | strong hip–knee coupling; deep knee flexion ~157° in transitions | [J Biomech Sci Eng 2015](https://www.jstage.jst.go.jp/article/jbse/10/4/10_15-00452/_article) |
| Smoothness | joints limited to ~600°/s (~20°/frame at 30 fps); noise smoothed more where dots are weak | engineering choice from the above |

Sources disagree on several numbers (zero-reference conventions, speed); the module uses round mid values and only
as a pull, never as a fixed pose.
