# Revisione della letteratura (aggiornamento 2026-11-06)

Stato: letture sul testo dei PDF caricati. Per ogni lavoro è indicato quali pagine sono state lette. Le parti non lette non entrano nel report.

## Lavori letti

### 1. X2CT-GAN — Ying et al., CVPR 2019, pp. 10619–10628
DOI 10.1109/CVPR.2019.01087. Letto: pp. 1–8.

- Ricostruisce un volume TC da due radiografie ortogonali con una GAN condizionale.
- Dati: 916 TC di training e 102 di test, scelte da 1.018 TC di LIDC-IDRI; volumi a 1 mm, 128³. Le radiografie sono DRR.
- Projection loss (eq. 4): il volume predetto viene proiettato in parallelo lungo tre piani ortogonali (assiale, coronale, sagittale) e confrontato in L1 con le proiezioni del volume reale. Non confronta le radiografie di ingresso: è un vincolo di forma, non di fedeltà alle radiografie.
- Pesi riportati: λ1 = 0,1; λ2 = λ3 = 10.
- Ablation (tab. 3, PSNR): GAN da sola 17,38 dB; con loss di ricostruzione 25,82; con projection loss 26,05; con il terzo termine 26,19. La projection loss aggiunge circa 0,2 dB.
- Ottimizzazione: Adam, lr 2e-4, 100 epoche.

### 2. DuoLift-GAN — Zhang e Ying, arXiv:2411.07941 (2024)
Letto: pp. 1–8, con tabelle e figure.

- Solleva le radiografie a 3D ripetendole lungo la direzione di proiezione, poi fonde due rami (feature e radiografie). Non usa geometria di proiezione esplicita.
- Loss: MSE e L1 sul volume, MSE e L1 sulla maschera polmonare (LungMask / TotalSegmentator), avversariale. Nessuna loss di riproiezione: la coerenza con le radiografie non è vincolata.
- Dati: stesso preprocessing di X2CT-GAN (916/102 TC), split interno 816 train / 100 validation, 128³, proiezioni con lo script ufficiale di X2CT-GAN.
- Training: GPU NVIDIA RTX A6000, 100 epoche, lr 5e-4 (CNN) e 1e-4 (GAN), batch 1 con accumulo su 4 passi.
- Doppia vista, tab. 1: PSNR(3D) 22,006 per DuoLift-GAN contro 21,577 per X2CT-GAN. SSIM(3D) 0,663 per la GAN contro 0,712 per la CNN. LPIPS migliore per la GAN (0,249 contro 0,276).
- Osservazione utile: la graduatoria cambia con la metrica. La CNN vince su PSNR e SSIM, la GAN su LPIPS. Per il nostro report questo rafforza la scelta di metriche per regione con bootstrap per paziente.
- Dice polmone (tab. 3): 0,931 per la CNN, 0,928 per la GAN.
- Limiti dichiarati dagli autori: nessun modello a diffusione per ragioni di memoria; LPIPS calcolato su immagini naturali.

### 3. XTransCT — Zhang et al., arXiv:2305.19621 (2023)
Letto: pp. 1–6 (introduzione, metodo, geometria di proiezione). Non letti: pagine 7 e seguenti, quindi la loss non è ancora verificata.

- Due radiografie a 45° e 135°. Dati: 50 pazienti di un ospedale, LIDC-IDRI, LNDb per la generalizzazione.
- Architettura: backbone 2D che produce feature 8×16×512, poi un transformer che, per ogni coordinata voxel normalizzata (x, y, z), restituisce il valore grigio. Nessuna convoluzione 3D.
- Tempo di ricostruzione dichiarato: 44 ms per volume, circa il 300% più veloce delle CNN 3D secondo gli autori.
- Le DRR sono descritte come raggi conici con attenuazione accumulata e interpolazione ai piani CT: stesso principio della nostra proiezione.

### 4. X-Recon — Wang et al., arXiv:2407.15356 (2024)
Letto: pp. 1–6. Dataset privato, non LIDC-IDRI.

- Dati: 255 pazienti con pneumotorace e 279 soggetti sani, Shanghai, 2015–2021. Radiografie PA e laterali ottenute da DRR. Split 4:1.
- Generatore: due encoder-decoder (PA e laterale) fusi da un modulo di rendering multiscala; discriminatore con CoordConv 3D; LSGAN.
- Loss: L2 di ricostruzione sul volume reale; L1 di riproiezione (ProST, proiezione differenziabile) su tre viste ortogonali (PA, assiale, laterale), con rotazioni e traslazioni casuali in training.
- Risoluzione: 224³ voxel, 1,6 mm di media.
- Valutazione a valle: Pearson 0,77 tra la percentuale di pneumotorace misurata sulla TC ricostruita e quella sulla TC reale (segmentazione zero-shot PTX-Seg).

### 5. Coarse-Fine CVAA-GAN — Qiao et al., arXiv:2408.09736 (2024)
Letto: pp. 1–4 (testo completo del metodo e dei risultati). Dataset privato.

- Dati: 268 TC lombari, 212 training e 56 test, ricampionate a 2 mm isotropi, crop 128³, DRR.
- Architettura: due encoder 2D paralleli; le feature 2D vengono ripetute in 3D lungo la direzione di proiezione (come DuoLift e X2CT-GAN); decoder con due moduli: View Attention Alignment (attenzione softmax tra le viste) e Fine Distillation (gate sigmoide), su più scale.
- Loss: LSGAN, L2 sul volume, e proiezione ortogonale in parallelo del volume predetto confrontata con le DRR del volume reale.
- Risultati su 56 test (tab. I): CVAA-GAN MAE 0,01635, PSNR 29,75, SSIM 0,8113. X2CT-GAN riprodotta dagli autori: MAE 0,01975, PSNR 27,84, SSIM 0,7673. PSR e 3DCNN sono più deboli.
- Training: Adam lr 2e-4, batch 4, 200 epoche, NVIDIA A40.
- Nota: gli autori hanno riprodotto X2CT-GAN sul proprio dataset, quindi i numeri non sono confrontabili con quelli pubblicati da Ying et al.

### 6. DPMA — Wu et al., Scientific Reports 15, 16894 (2025)
DOI 10.1038/s41598-025-02133-5. Letto: pp. 1–6 (metodo fino all'eq. 21). Licenza CC BY-NC-ND: si cita, non si riproducono le figure.

- Problema diverso: sparse-view CT su una sola geometria, non radiografie biplanari.
- Rilevante per il vincolo fisico: il modulo PIC applica una correzione di consistenza dei dati (un passo di tipo Landweber, x̂ − Aᵀ(Ax̂ − s)) e la rete che segue viene addestrata con loss sul volume reale.
- Differenza dal nostro modello C: DPMA inserisce la consistenza nell'architettura, con un passo esplicito che usa l'operatore di proiezione; C la impone con una loss di riproiezione durante il training.

## Lavori non letti

- **XTransCT**: pagine 7 e seguenti. Da leggere per la loss e la valutazione.
- **SA-LDM**: nessun PDF caricato. Modello a diffusione latente per radiografie biplanari; anno e sede da confermare.

## Implicazioni per il report

- Nei lavori letti, la projection loss confronta le proiezioni del volume predetto con quelle del volume reale. Nessuno, tra quelli letti, confronta il volume predetto con le radiografie di ingresso con geometria a fascio conico. Questa differenza è il punto di posizionamento del nostro modello C, ma va confermata su XTransCT e SA-LDM prima di scriverla nel report.
- I numeri non sono confrontabili tra lavori: cambiano dataset, risoluzione (1 mm, 2 mm, 1,6 mm, 6 mm del nostro pilota), metriche e riproduzioni di baseline.
- Il nostro contributo da dichiarare è la valutazione per regione anatomica con intervallo di confidenza per paziente e la misura della coerenza di riproiezione, non un nuovo modello.
- Limite da dichiarare: 64³ a 6 mm, contro 128³ a 1 mm nei lavori su LIDC-IDRI.

## Da fare

- Leggere le pagine restanti di XTransCT (loss e valutazione).
- Confermare anno e sede di SA-LDM, con il PDF o la pagina dell'autore.
- Cercare studi di radioterapia che usano ricostruzione da radiografie, per il contesto clinico.
- Verificare la licenza di LIDC-IDRI (task 1.1, a carico dell'utente).
