# Revisione della letteratura (primo passo, 2026-11-06)

Stato: ricerca preliminare su abstract e pagine di indice. I testi completi non sono ancora stati letti: prima di usare un dato nel report va verificato sul PDF.

## Lavori centrali

1. **X2CT-GAN** — Ying, Guo, Ma, Wu, Weng, Zheng. CVPR 2019, pp. 10619–10628. DOI 10.1109/CVPR.2019.01087.
   - Ricostruisce una TC da due radiografie ortogonali con una GAN; addestrata su radiografie sintetiche (DRR) ricavate da TC toraciche pubbliche.
   - Loss: MSE e avversariale nell'abstract; la figura 2 riporta anche una *projection loss* (PL), cioè il vincolo di riproiezione del nostro modello C. Da verificare nel testo quale peso e quale formulazione usano.
   - Motiva i dati sintetici con la scarsità di coppie reali radiografia–TC.

2. **DuoLift-GAN** — arXiv:2411.07941. Ricostruzione da vista singola e biplanare, su LIDC-IDRI. Solleva le immagini 2D e le feature a 3D prima della ricostruzione: concettualmente vicino alla nostra retroproiezione del modello A.

3. **XTransCT** — Zhang et al., arXiv:2305.19621. Ricostruzione da due radiografie ortogonali con un transformer; dati DRR da 50 pazienti.

4. **X-Recon** — arXiv:2407.15356. Ricostruzione da radiografie ortogonali, con valutazione anche su segmentazione dello pneumotorace; utile come riferimento per la valutazione clinica futura.

5. **Coarse-Fine View Attention GAN** — arXiv:2408.09736. Biplanare, con attenzione tra le due viste nella fusione delle feature.

6. **SA-LDM** — modello a diffusione latente per la ricostruzione biplanare (pagina dell'università di Pechino; anno e sede da confermare).

## Riferimento per il vincolo fisico

7. **DPMA** — Wu et al., Scientific Reports 15, 16894 (2025). DOI 10.1038/s41598-025-02133-5. Sparse-view CT con un modulo di coerenza fisica sui dati di proiezione. Licenza CC BY-NC-ND: va citato, non riprodotto.

## Ipotesi di lavoro per il report

- Il nostro confronto A vs C è vicino a quello di X2CT-GAN (con e senza projection loss), ma con una geometria esplicita e una valutazione su regioni anatomiche separate.
- Il contributo da dichiarare è la valutazione per regione e la coerenza di riproiezione, non la novità del metodo.

## Da fare

- Leggere i testi completi di X2CT-GAN e DuoLift-GAN, con attenzione a come definiscono la projection loss e i dati di test.
- Cercare studi su radioterapia con ricostruzione da radiografie, per collegare il lavoro al contesto clinico.
- Confermare anno e sede di SA-LDM.
