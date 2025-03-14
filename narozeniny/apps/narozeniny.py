# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair==5.5.0",
#     "marimo",
#     "numpy==2.2.3",
#     "polars==1.24.0",
# ]
# [tool.marimo.display]
# theme = "dark"
# ///

import marimo

__generated_with = "0.11.5"
app = marimo.App(width="medium", layout_file="layouts/narozeniny.grid.json")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import numpy as np
    import altair as alt
    import polars as pl
    import pickle
    import requests
    import imageio
    return alt, imageio, np, pickle, pl, requests


@app.cell
def _(mo):
    done_button = mo.ui.run_button(label='HOTOVO?')
    password = mo.ui.text(placeholder='Heslo?')
    return done_button, password


@app.cell
def _(mo):
    results = (
        mo.md(
            '''
        **Jaké to bylo?**
        {riddle_1}

        **Kolik je na fotce koček?**
        {riddle_2}

        **Kdo jsi?**
        {riddle_3}

        **Co je na obrázku?**
        {riddle_4}

        **Hmmmm?**
        {riddle_5}
        '''
        )
        .batch(
            riddle_1=mo.ui.text(),
            riddle_2=mo.ui.number(value=0),
            riddle_3=mo.ui.text(),
            riddle_4=mo.ui.text(),
            riddle_5=mo.ui.text(),
        )
        .form(submit_button_label='Check')
    )
    return (results,)


@app.cell
def _(mo):
    correct = mo.md(f"# {mo.icon('fxemoji:whiteheavycheckmark')}")
    incorrect = mo.md(f"# {mo.icon('emojione:cross-mark-button')}")
    return correct, incorrect


@app.cell
def _(cats_2, crossroad, jiggsaw, mo, nonogram, wordle):
    problems = mo.accordion(
        {
            "Hádanka 1": nonogram,
            "Hádanka 2": cats_2,
            "Hádanka 3": wordle,
            "Hádanka 4": jiggsaw,
            "Hádanka 5": mo.md(f'{crossroad}\n1-2 9-5 11-3 6-4 8-9 &nbsp; &nbsp; 11-5 7-6 3-1 3-\"7-8\" 1-6 6-5!'),
        }
    )
    return (problems,)


@app.cell
def _(np):
    puzzles = np.zeros(5)
    return (puzzles,)


@app.cell
def _(mo, puzzles, results):
    n_correct = 0
    mo.stop(not results.value)

    try:
        results.value
        if results.value['riddle_1'].startswith('legend'):
            puzzles[0] = 1
        else:
            puzzles[0] = 0

        if results.value['riddle_2'] == 7:
            puzzles[1] = 1
        else:
            puzzles[1] = 0

        if results.value['riddle_3'].startswith('karel'):
            puzzles[2] = 1
        else:
            puzzles[2] = 0

        if results.value['riddle_4'].startswith('bebe'):
            puzzles[3] = 1
        else:
            puzzles[3] = 0

        if results.value['riddle_5'].startswith(('pořád dvacet', 'porad dvacet')):
            puzzles[4] = 1
        else:
            puzzles[4] = 0
    except:
        pass

    n_correct = puzzles.sum()


    if n_correct == 0:
        keep_probability = 0
    if n_correct == 1:
        keep_probability = 0.001
    if n_correct == 2:
        keep_probability = 0.003
    if n_correct == 3:
        keep_probability = 0.008
    if n_correct == 4:
        keep_probability = 0.05
    if n_correct == 5:
        keep_probability = 1
    return keep_probability, n_correct


@app.cell
def _(correct, incorrect, puzzles, results):
    results.value

    table_string = """
    | Hádanka | Výsledek |
    |--|:--:|"""
    for i in range(5):
        table_string += f'\n| Hádanka {i+1} | {correct if puzzles[i] != 0 else incorrect} |'
    return i, table_string


@app.cell
def _(mo):
    path_pend = str(mo.notebook_location() / 'public' / 'pend.png')
    return (path_pend,)


@app.cell
def _(imageio, keep_probability, mo, np, path_pend, pickle, requests):
    try:
        response = requests.get(path_pend)
        image = imageio.imread(response.content)
        pend = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8).copy()
    except:
        with open(mo.notebook_location() / 'public' / 'pend.pkl', 'rb') as f:
            pend = pickle.load(f)

    random_mask = np.random.rand(*pend.shape) > keep_probability
    hide_mask = random_mask
    pend[hide_mask] = 0
    return f, hide_mask, image, pend, random_mask, response


@app.cell
def _(mo):
    iframe = f"""<iframe
      srcdoc='
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="UTF-8">
          <title>Embedded Widget</title>
        </head>
        <body>
          <div class="widget-container" data-widget-id="nonograms" data-widget-height data-widget-rnd="1" data-widget-lang="en"></div>
          <script async src="https://cdn.player.zone/static/embed.js?nocache=707516"></script>
        </body>
        </html>
    </iframe>"""

    nonogram = mo.iframe(
        iframe, height="700px"
    )
    return iframe, nonogram


@app.cell
def _(mo):
    iframe2 = """<iframe src="https://wordly.org/cs?challenge=a2FyZWw" width="100%" height="680" frameborder="0" scrolling="no"></iframe>
    """

    wordle = mo.iframe(
        iframe2, height="700px"
    )
    return iframe2, wordle


@app.cell
def _(mo):
    cats_2 = mo.image(src='./public/cats.png')
    return (cats_2,)


@app.cell
def _(mo):
    iframe3 = """<iframe src="https://www.jigidi.com/s/3mt9ni/?embed" title="Jigsaw puzzle" style="width:100%;height:575px" frameborder="0" allow="fullscreen" allowfullscreen></iframe>
    """

    jiggsaw = mo.iframe(
        iframe3, height="700px"
    )
    return iframe3, jiggsaw


@app.cell
def _(mo):
    iframe4 = """<iframe width="700" height="700" style="border:3px solid black; margin:auto; display:block" frameborder="0" src="https://crosswordlabs.com/embed/krizovka-3-2?clue_height=42"></iframe>
    """

    crossroad = mo.iframe(
        iframe4, height="730px"
    )
    return crossroad, iframe4


@app.cell
def _(mo):
    mo.md(r"""# K dárku musíš najít heslo!""")
    return


@app.cell
def _(problems):
    problems
    return


@app.cell
def _(mo, table_string):
    mo.md(table_string)
    return


@app.cell
def _(results):
    results
    return


@app.cell
def _(mo, pend):
    mo.image(pend)
    return


@app.cell
def _(done_button, mo, n_correct, results):
    results.value

    if n_correct == 5:
        mo.output.replace(done_button)
    else:
        mo.output.replace(mo.md(''))
    return


@app.cell
def _(done_button, mo, password, results):
    results.value

    if done_button.value:
        mo.output.replace(password)
    else:
        mo.output.replace(mo.md('Hmmmmmmmm...'))

    if password.value.lower() == 'pendulum':
        mo.output.replace(mo.md('[Tak už balíš?](https://tribes-unite.com/)'))
    return


@app.cell
def _(mo, password):
    if password.value.lower() == 'pendulum':
        mo.output.replace(mo.md('[Tak už balíš?](https://tribes-unite.com/)'))
    else:
        mo.output.replace(mo.md(''))
    return


if __name__ == "__main__":
    app.run()
