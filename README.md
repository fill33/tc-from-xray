# tcrec — Ricostruzione di una TC toracica da due radiografie

Progetto di ricerca (portfolio) sul problema di stimare un volume TC 3D a partire da due radiografie ortogonali del torace, con deep learning e un vincolo fisico di riproiezione.

Il design e il piano pre-registrato sono in `input/ideas/`. Le ipotesi e i criteri di falsificazione sono congelati nel piano.

## Stato

| Componente | Stato |
|---|---|
| Geometria di proiezione a fascio conico (proiezione e retroproiezione) | implementata, con test |
| Modello A: retroproiezione + U-Net 3D | implementato, con test |
| Modello C: loss di riproiezione | implementata, con test |
| Pipeline dati (DICOM, annotazioni, DRR su dataset reale) | da fare, richiede il sottoinsieme LIDC-IDRI |
| Split per paziente | da fare (Task 2.5) |
| Addestramento e valutazione sul test set | da fare, su GPU cloud |

## Limiti dichiarati

- Le radiografie di addestramento sono **DRR simulate** dalle TC: il *domain gap* con radiografie reali non è stimato in questa versione.
- Il valore di `mu_water` in `src/tcrec/hu.py` è indicativo e va verificato su NIST XCOM per l'energia del fascio.
- Il progetto non è uno strumento clinico: non calcola la dose e non va usato su pazienti.

## Installazione

```bash
pip install -e ".[dev]"
pytest
```

## Struttura

```
src/tcrec/geometry.py   proiezione e retroproiezione a fascio conico (differenziabili)
src/tcrec/models.py     U-Net 3D e modello A
src/tcrec/losses.py     loss di volume e di riproiezione; loss del modello C
src/tcrec/hu.py         conversione HU <-> coefficiente di attenuazione
tests/                  test di geometria, loss e modelli su dati sintetici
```

## Licenza

Da definire prima della pubblicazione. I dati LIDC-IDRI hanno termini d'uso propri, da rispettare.
