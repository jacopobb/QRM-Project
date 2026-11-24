# Report LaTeX – guida passo passo (VS Code + GitHub, Mac)

Il report è un unico documento, `main.tex`, che richiama le sezioni in `sections/` con `\input`.
Ognuno scrive **solo nei propri file** (vedi `homework2/README.md` §5, membri A, B, C, D);
compilando `main.tex` si ottiene il PDF completo. Nessuno deve unire file a mano.

```
homework2/report/
├── main.tex            documento principale: pacchetti, titolo, ordine delle sezioni (lo modifica solo D)
├── sections/           una sezione per file: QUI si scrive
├── references.bib      bibliografia
├── make_tables.py      genera tables/ e figures/ dai risultati del codice (non toccare i file generati)
├── tables/  figures/   creati da main_hw2.py: compaiono dopo aver lanciato il codice
└── README.md           questa guida
```

---

## 1. Installazione (una volta sola, ognuno sul proprio Mac)

**1.1 LaTeX (MacTeX).** Scarica `MacTeX.pkg` da <https://www.tug.org/mactex/> e installalo (circa 6 GB,
richiede un po' di tempo). In alternativa, con Homebrew: `brew install --cask mactex-no-gui`.
Poi **chiudi del tutto VS Code** (Cmd+Q) e riaprilo, e nel terminale verifica:
```bash
which latexmk        # deve rispondere /Library/TeX/texbin/latexmk
```

**1.2 Estensione di VS Code.** In VS Code apri Estensioni (Cmd+Shift+X), cerca **LaTeX Workshop**
(autore James Yu) e clicca *Install*. Non serve configurare niente.

**1.3 Prova.** Apri in VS Code la cartella `QRM-Project`, poi il file `homework2/report/main.tex`:
- **Cmd+Option+B** compila (oppure l'icona ▷ "Build LaTeX project" in alto a destra);
- **Cmd+Option+V** apre il PDF in una scheda accanto al testo.

Se tabelle e figure non sono ancora state generate, al loro posto compaiono riquadri rossi
"not generated yet": è normale, il PDF si compila lo stesso.

---

## 2. Come si scrive

- Apri il **tuo** file in `sections/` (es. `04_backtesting.tex`) e scrivi; a ogni **salvataggio (Cmd+S)**
  VS Code ricompila tutto il report (la prima riga `% !TEX root = ../main.tex` gli dice qual è il documento principale).
- Il PDF si aggiorna da solo; **Cmd+clic** su un punto del PDF porta alla riga corrispondente del `.tex`.
- I testi in rosso `\todo{...}` indicano cosa commentare: sostituiscili con il tuo testo e cancellali.

Comandi LaTeX più utili:

| Cosa | Come si scrive |
|---|---|
| nuovo paragrafo | una riga vuota |
| grassetto / corsivo | `\textbf{testo}` / `\emph{testo}` |
| formula nel testo / a parte | `$\VaR_t^\alpha$` / `\begin{equation} ... \end{equation}` |
| citare un articolo | `\citep{kupiec1995}` (la chiave è in `references.bib`) |
| richiamare tabella / figura / sezione | `Table~\ref{tab:bt_EWP_0.01}`, `Figure~\ref{fig:var_SMI}`, `Section~\ref{sec:es}` |
| elenco puntato | `\begin{itemize} \item primo \item secondo \end{itemize}` |
| commento (non stampato) | `% testo` |
| caratteri speciali | scrivi `\%`, `\&`, `\_`, `\$` (senza `\` danno errore) |

Le etichette (`\label{...}`) delle tabelle sono nei file in `tables/`: per vederle apri il file, es.
`tables/tab_bt_EWP_0.01.tex`, riga `\label{tab:bt_EWP_0.01}`.
Per aggiungere un articolo alla bibliografia: copia una voce in `references.bib`, cambia chiave e dati, poi citala.

---

## 3. Flusso di lavoro quotidiano con GitHub

Dal terminale di VS Code, nella cartella `QRM-Project`:

```bash
# 1. PRIMA di iniziare: scarica il lavoro degli altri
git pull

# 2. scrivi nel tuo file, salva, controlla che il PDF si compili

# 3. QUANDO hai finito (anche solo per oggi): carica SOLO i tuoi file
git add homework2/report/sections/04_backtesting.tex
git commit -m "Backtesting section: comment on EWP"
git push
```

Regole per non creare conflitti:
1. **Sempre `git pull` prima di iniziare** e prima di `git push`.
2. **Modifica solo i tuoi file.** Se serve una modifica a `main.tex`, `00_roles.tex` o al file di un altro, chiedila a chi ne è responsabile.
3. **Non usare `git add .`**: aggiungi i file per nome. Il PDF (`main.pdf`) e i file temporanei di LaTeX
   (`.aux`, `.log`, ...) sono esclusi da `.gitignore` e non vanno caricati.
4. `tables/` e `figures/` li carica **solo D**, dopo aver lanciato `python main_hw2.py` con i dati veri.
5. Fai commit piccoli e frequenti, con messaggi che dicano cosa hai cambiato.

---

## 4. Problemi frequenti

| Problema | Soluzione |
|---|---|
| `git push` rifiutato ("rejected ... fetch first") | qualcuno ha caricato prima di te: `git pull`, poi `git push` |
| `git pull` segnala un **conflitto** | succede solo se due persone hanno modificato lo stesso file: apri il file, VS Code mostra le due versioni con i pulsanti *Accept Current / Incoming / Both*; scegli, salva, poi `git add <file>`, `git commit`, `git push` |
| errore di compilazione | in VS Code apri *Problems* (Cmd+Shift+M): indica file e riga. Le cause tipiche: `%`, `&`, `_` senza `\`, una `{` o `$` non chiusa |
| `latexmk` non trovato | MacTeX non installato o VS Code non riavviato dopo l'installazione (Cmd+Q e riapri) |
| riferimento `??` nel PDF | etichetta scritta male, oppure serve una seconda compilazione: salva di nuovo |
| il PDF non si aggiorna | Cmd+Option+B per ricompilare a mano |
| il PDF non si compila e non si capisce perché | dal terminale: `cd homework2/report && latexmk -pdf main.tex`; per ripartire pulito: `latexmk -C` |

---

## 5. Alla fine (D)

1. Mettere i dati veri in `homework2/data/` e lanciare `cd homework2 && python main_hw2.py` (rigenera risultati, tabelle e figure).
2. `git add homework2/report/tables homework2/report/figures`, commit e push.
3. Controllare che nel PDF non ci siano più testi rossi `TODO` né riquadri "not generated yet".
4. Compilare `main.tex` e consegnare `main.pdf` (non è su GitHub: si carica direttamente dove richiesto dal prof).
