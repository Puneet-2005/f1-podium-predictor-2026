# F1 2026 Race Predictor

A Python project for exploring Formula 1 predictions. It makes an early winner pick before qualifying, ranks podium contenders once the grid is available, and compares its picks with official race results. Everything runs in the terminal.

## Get started

Clone the repository and enter its folder in PowerShell.

```powershell
git clone https://github.com/Puneet-2005/f1-podium-predictor-2026.git
Set-Location f1-podium-predictor-2026
```

Use Python 3.10 or later. Open the cloned project folder in VS Code and start a PowerShell terminal there. The folder should contain `predict.py` and `requirements.txt`.

Set up the project once:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run a historical replay first. This works with the included race dataset:

```powershell
.\.venv\Scripts\python.exe predict.py --race brazil --dashboard
```

The dashboard shows the predicted order, starting grid, actual finishes, and average position error. The command prints the path to the full CSV.

## Forecast a race

| What you want | Script | What you need |
| --- | --- | --- |
| An early winner pick | `winner_forecast.py` | Published driver roster and earlier race results |
| Three podium picks | `predict.py --model podium` | The race grid and qualifying data, with optional practice pace |
| A forecast you can score later | `live_podium.py capture` | A published starting grid before race start |
| A comparison across past races | `compare_models.py` | Historical results and the practice cache if enabled |

If your terminal prompt already shows `(.venv)`, use the shorter commands below. To activate the environment in PowerShell, run `.\.venv\Scripts\Activate.ps1`. You can also use `.\.venv\Scripts\python.exe` in place of `python`.

The Las Vegas command is a pre-qualifying example. For a future weekend, use the corresponding race slug while its pre-qualifying window is open.

```powershell
python winner_forecast.py --race las-vegas --dashboard
python predict.py --race mexico --model podium --dashboard
python predict.py --backtest
```

Use `python predict.py --list` to see the available race names. The project uses `brazil` for São Paulo and `great-britain` for Silverstone.

### Add practice pace

Populate the practice cache and run the podium model with practice pace enabled:

```powershell
python practice_pace.py --update
python predict.py --race mexico --model podium --practice-pace --dashboard
```

The dashboard shows the practice pace gap and how many laps contributed. A negative gap means the driver was faster than comparable peers. A missing gap means the data did not meet the comparison rules. FIA component disclosures appear alongside predictions as weekend context.

### Save a live podium forecast

After the starting grid is published and before the race starts, run:

```powershell
python live_podium.py capture --race las-vegas --output predictions/2026/live
```

Keep the snapshot file printed by the command. It records the three sets of picks, practice coverage, model settings, and UTC capture time. After the official race result is published, score that saved file:

```powershell
python live_podium.py score --snapshot "<path to the saved snapshot file>"
```

The scorer counts how many of the three picks made the podium in any order.

## How the models work

- **Ridge regression** estimates finishing order from the grid, qualifying pace, and recent form. The starting-grid baseline assumes every driver finishes in the position they started.
- **Logistic regression for podiums** scores each driver on the likelihood of a top-three finish. The three highest scores become the picks. Optional practice pace adds comparable long-run lap times.
- **Pre-qualifying winner forecast** uses earlier rounds to rank potential winners before the grid is published. The recent-form baseline picks the driver with the best average finish across their last three races.

The backtest trains on earlier races and evaluates each subsequent race in order. On the 17-race development replay, the practice-pace podium model picked **32 of 51** podium drivers. The same model without practice pace picked **28 of 51**, and the grid picked **29 of 51**. Usable practice data covered **194 of 374** driver entries. The early winner model picked **4 of 17** winners, while the recent-form baseline picked **7 of 17**. These are historical scores. Dashboard model scores have not been calibrated as probabilities.

## Data and tests

- [Formula 1 race results](https://www.formula1.com/en/results/2026/races) for official grids and finishes.
- [OpenF1](https://openf1.org/docs/) for historical timing, tyres, and race-control data.
- [FIA decision documents](https://www.fia.com/documents/championships/110/2026) for car presentation disclosures.

Run the test suite with `python -m unittest discover -v` from the project folder. Generated predictions and local caches belong on your machine. The project includes the small race and FIA data files needed for the initial examples.
