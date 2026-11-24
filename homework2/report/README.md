# Report LaTeX – guida passo passo (VS Code + GitHub, Mac e Windows)

Il report è un unico documento, `main.tex`, che richiama le sezioni in `sections/` con `\input`.
Ognuno scrive **solo nei propri file** (vedi `homework2/README.md` §5, membri A, B, C, D);
compilando `main.tex` si ottiene il PDF completo. Nessuno deve unire file a mano.

La procedura è la stessa su Mac e su Windows (es. un portatile Lenovo): cambiano solo **l'installazione di LaTeX**
e **le scorciatoie da tastiera** (Cmd sul Mac, Ctrl su Windows). Le differenze sono indicate in ogni passo.

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

## 1. Installazione (una volta sola, ognuno sul proprio computer)

### 1.1 LaTeX

**Mac – MacTeX.** Scarica `MacTeX.pkg` da <https://www.tug.org/mactex/> e installalo (circa 6 GB).
È un pacchetto `.pkg`: non va trascinato in Applicazioni, si installa da solo. In alternativa, con Homebrew:
`brew install --cask mactex-no-gui`.

**Windows – TeX Live (consigliato).** Scarica `install-tl-windows.exe` da <https://tug.org/texlive/windows.html>,
eseguilo e scegli *Install* con le opzioni proposte (installazione completa, alcuni GB: può richiedere 30–60 minuti).
TeX Live contiene già Perl, che serve a `latexmk`.

> Alternativa su Windows: **MiKTeX** (<https://miktex.org>), più leggero, scarica i pacchetti man mano. In questo caso
> devi installare anche **Strawberry Perl** (<https://strawberryperl.com>), altrimenti `latexmk` non funziona, e nella
> console di MiKTeX impostare *Install missing packages on-the-fly: Yes*. Se non hai motivi particolari, usa TeX Live.

### 1.2 Git (solo Windows, se non è già installato)
Su Mac git c'è già. Su Windows apri il terminale di VS Code e scrivi `git --version`: se dà errore, installa
**Git for Windows** da <https://git-scm.com/download/win> (opzioni proposte) e riavvia VS Code.

### 1.3 Verifica
**Chiudi del tutto VS Code** (Mac: Cmd+Q; Windows: chiudi tutte le finestre) e riaprilo, poi nel terminale:

| | Comando | Risposta attesa |
|---|---|---|
| Mac | `which latexmk` | `/Library/TeX/texbin/latexmk` |
| Windows | `latexmk -v` | una riga `Latexmk, John Collins, ... Version ...` |

Se non trovi il comando: riapri VS Code o riavvia il computer; su Windows con MiKTeX controlla di aver installato Perl.

### 1.4 Estensione di VS Code (uguale per tutti)
Apri Estensioni (Mac: Cmd+Shift+X; Windows: Ctrl+Shift+X), cerca **LaTeX Workshop** (autore James Yu) e clicca
*Install*. Non serve configurare niente.

### 1.5 Prova
Apri in VS Code la cartella `QRM-Project`, poi il file `homework2/report/main.tex`:

| Azione | Mac | Windows |
|---|---|---|
| compila il report | Cmd+Option+B | Ctrl+Alt+B |
| apri il PDF accanto al testo | Cmd+Option+V | Ctrl+Alt+V |
| dal PDF alla riga del `.tex` | Cmd+clic sul PDF | Ctrl+clic sul PDF |
| salva (e ricompila) | Cmd+S | Ctrl+S |
| pannello errori (*Problems*) | Cmd+Shift+M | Ctrl+Shift+M |

In alternativa a tutte le scorciatoie: l'icona ▷ "Build LaTeX project" in alto a destra.
Se tabelle e figure non sono ancora state generate, al loro posto compaiono riquadri rossi
"not generated yet": è normale, il PDF si compila lo stesso.

---

## 2. Come si scrive (uguale per tutti)

- Apri il **tuo** file in `sections/` (es. `04_backtesting.tex`) e scrivi; a ogni **salvataggio** VS Code ricompila
  tutto il report (la prima riga `% !TEX root = ../main.tex` gli dice qual è il documento principale).
- Il PDF si aggiorna da solo; Cmd+clic (Mac) o Ctrl+clic (Windows) su un punto del PDF porta alla riga del `.tex`.
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

## 3. Flusso di lavoro quotidiano con GitHub (comandi identici su Mac e Windows)

Dal terminale di VS Code, nella cartella `QRM-Project` (su Mac il terminale è zsh, su Windows PowerShell:
i comandi git sono gli stessi):

```bash
# 1. PRIMA di iniziare: scarica il lavoro degli altri
git pull

# 2. scrivi nel tuo file, salva, controlla che il PDF si compili

# 3. QUANDO hai finito (anche solo per oggi): carica SOLO i tuoi file
git add homework2/report/sections/04_backtesting.tex
git commit -m "Backtesting section: comment on EWP"
git pull
git push
```

- `git push` da solo non carica niente: servono prima `git add` (quali file) e `git commit` (il salvataggio con un messaggio).
- `git status` mostra quali file hai modificato: usalo prima di `git add`.
- Su Windows i percorsi si possono scrivere sia con `/` sia con `\` (es. `homework2\report`).

Regole per non creare conflitti:
1. **Sempre `git pull` prima di iniziare** e prima di `git push`.
2. **Modifica solo i tuoi file.** Se serve una modifica a `main.tex`, `00_roles.tex` o al file di un altro, chiedila a chi ne è responsabile.
3. **Non usare `git add .`**: aggiungi i file per nome. Il PDF (`main.pdf`) e i file temporanei di LaTeX
   (`.aux`, `.log`, ...) sono esclusi da `.gitignore` e non vanno caricati, come `__pycache__` e `.venv`.
4. `tables/` e `figures/` li carica **solo D**, dopo aver lanciato `python main_hw2.py` con i dati veri.
5. Fai commit piccoli e frequenti, con messaggi che dicano cosa hai cambiato.

---

## 4. Problemi frequenti

| Problema | Soluzione |
|---|---|
| `git push` rifiutato ("rejected ... fetch first") | qualcuno ha caricato prima di te: `git pull`, poi `git push` |
| `git pull` segnala un **conflitto** | succede solo se due persone hanno modificato lo stesso file: apri il file, VS Code mostra le due versioni con i pulsanti *Accept Current / Incoming / Both*; scegli, salva, poi `git add <file>`, `git commit`, `git push` |
| `git commit`: "Please tell me who you are" | una volta sola: `git config --global user.name "Nome Cognome"` e `git config --global user.email "email@github"` |
| errore di compilazione | apri *Problems* (Cmd/Ctrl+Shift+M): indica file e riga. Le cause tipiche: `%`, `&`, `_` senza `\`, una `{` o `$` non chiusa |
| `latexmk` non trovato | LaTeX non installato o VS Code non riavviato dopo l'installazione |
| (Windows, MiKTeX) errore su **Perl** | installa Strawberry Perl e riavvia VS Code, oppure usa TeX Live |
| (Windows, MiKTeX) finestra "install package" | clicca *Install*: MiKTeX scarica il pacchetto mancante |
| (Windows) avviso "LF will be replaced by CRLF" | innocuo: git converte solo i fine riga |
| riferimento `??` nel PDF | etichetta scritta male, oppure serve una seconda compilazione: salva di nuovo |
| il PDF non si aggiorna | ricompila a mano (Cmd+Option+B / Ctrl+Alt+B) |
| il PDF non si compila e non si capisce perché | dal terminale: `cd homework2/report` e `latexmk -pdf main.tex`; per ripartire pulito: `latexmk -C` |

---

## 5. Alla fine (D)

1. Mettere i dati veri in `homework2/data/` e lanciare `cd homework2` e `python main_hw2.py` (rigenera risultati,
   tabelle e figure; su Windows, se `python` non viene trovato, usa `py main_hw2.py`).
2. `git add homework2/report/tables homework2/report/figures`, commit e push.
3. Controllare che nel PDF non ci siano più testi rossi `TODO` né riquadri "not generated yet".
4. Compilare `main.tex` e consegnare `main.pdf` (non è su GitHub: si carica direttamente dove richiesto dal prof).
