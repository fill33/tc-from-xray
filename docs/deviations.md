# Registro delle deviazioni dal piano

Piano di riferimento: `input/ideas/ricostruzione-tc-da-radiografie-2d-plan.md` (pre-registrato il 2026-11-04).

| Data | Deviazione | Motivo | Effetto sui risultati |
|---|---|---|---|
| 2026-11-05 | Campione pilota di 40 pazienti (LIDC-IDRI-0001..0040) invece di 200–300 | Il dataset completo non è scaricabile dalla sessione; primo test della pipeline | Risultati solo indicativi; intervalli di confidenza ampi |
| 2026-11-05 | Segmentazione polmonare con soglia sulle HU (-400) invece di un segmentatore pre-addestrato | Pilota senza dipendenze esterne; il segmentatore va scelto e documentato (task 2.4) | Metrica Dice da considerare provvisoria |
| 2026-11-05 | Risoluzione 64³ a 6 mm per voxel, rivelatore 64 px a 10 mm | Memoria e tempi sulla GPU del pilota | Dettagli fini (noduli piccoli) non ricostruibili; va dichiarato nel report |
| 2026-11-05 | Checkpoint scelto sull'errore di volume L1 del validation set, per A e C | Criterio identico per entrambi, indipendente da lambda | Nessuno sul confronto; registrato per trasparenza |

Lo split del pilota (`split_pilot40.json`) non è congelato: va rigenerato se il campione cambia.

## Risultato provvisorio del pilota (40 pazienti, 30 epoche, test 6 pazienti)

- Differenza A − C, MAE in HU, bootstrap 95% per paziente:
  - polmone: +26,8 [−26,4; +66,9]: inconcludente;
  - tessuto molle (mediastino): −103,6 [−117,5; −89,3]: A migliore di C.
- Ipotesi pre-registrata non supportata in questo pilota. Riportata senza modifiche.
- Decisione: aumentare il campione (pazienti da 0041 in poi) prima di qualsiasi conclusione.
