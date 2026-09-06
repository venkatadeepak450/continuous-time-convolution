let pyodide = null;

let currentImage = null;


const convolveBtn =
    document.getElementById("convolveBtn");

const downloadBtn =
    document.getElementById("downloadBtn");

const statusText =
    document.getElementById("status");

const figureContainer =
    document.getElementById("figureContainer");

const valuesBox =
    document.getElementById("values");


async function loadPython() {

    statusText.textContent =
        "Loading Python, NumPy, SymPy and Matplotlib...";

    convolveBtn.disabled = true;


    try {

        pyodide = await loadPyodide();


        await pyodide.loadPackage([
            "numpy",
            "sympy",
            "matplotlib"
        ]);


        statusText.textContent = "Ready.";

        convolveBtn.disabled = false;


    } catch (error) {

        statusText.textContent =
            "Could not load the Python environment.";

        console.error(error);
    }
}



function getInput(id) {

    return document
        .getElementById(id)
        .value
        .trim();

}



async function convolve() {

    if (!pyodide) {
        return;
    }


    const xExpr =
        getInput("xFunction");

    const hExpr =
        getInput("hFunction");

    const linspaceExpr =
        getInput("linspace");


    convolveBtn.disabled = true;

    downloadBtn.disabled = true;


    statusText.textContent =
        "Calculating convolution...";


    try {

        pyodide.globals.set(
            "x_input",
            xExpr
        );

        pyodide.globals.set(
            "h_input",
            hExpr
        );

        pyodide.globals.set(
            "linspace_input",
            linspaceExpr
        );


        const result =
            await pyodide.runPythonAsync(`

import sympy as sp
import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import base64

from io import BytesIO


# --------------------------------------------------
# Symbols
# --------------------------------------------------

t = sp.symbols("t", real=True)

tau = sp.symbols("tau", real=True)


# --------------------------------------------------
# SymPy functions allowed in user input
# --------------------------------------------------

local_dict = {

    "sp": sp,

    "t": t,

    "pi": sp.pi,

    "E": sp.E,

    "sin": sp.sin,

    "cos": sp.cos,

    "tan": sp.tan,

    "asin": sp.asin,

    "acos": sp.acos,

    "atan": sp.atan,

    "sinh": sp.sinh,

    "cosh": sp.cosh,

    "tanh": sp.tanh,

    "exp": sp.exp,

    "sqrt": sp.sqrt,

    "log": sp.log,

    "ln": sp.log,

    "Abs": sp.Abs,

    "sign": sp.sign,

    "Heaviside": sp.Heaviside
}


# --------------------------------------------------
# Parse x(t) and h(t)
# --------------------------------------------------

x = sp.sympify(
    x_input,
    locals=local_dict
)

h = sp.sympify(
    h_input,
    locals=local_dict
)


# Only t is allowed as the variable

if x.free_symbols - {t}:

    raise ValueError(
        "Only t may be used as a signal variable."
    )


if h.free_symbols - {t}:

    raise ValueError(
        "Only t may be used as a signal variable."
    )


# --------------------------------------------------
# Parse np.linspace(...)
# --------------------------------------------------

import re


match = re.fullmatch(

    r"\\s*np\\.linspace\\(\\s*(.+?)\\s*,\\s*(.+?)\\s*,\\s*(\\d+)\\s*\\)\\s*",

    linspace_input

)


if not match:

    raise ValueError(
        "Linspace must be in the form "
        "np.linspace(start, stop, points)"
    )


start_expr = sp.sympify(
    match.group(1),
    locals=local_dict
)


stop_expr = sp.sympify(
    match.group(2),
    locals=local_dict
)


start = float(start_expr)

stop = float(stop_expr)

N = int(match.group(3))


if N < 2:

    raise ValueError(
        "Linspace must contain at least 2 points."
    )


if stop <= start:

    raise ValueError(
        "Linspace stop must be greater than start."
    )


# --------------------------------------------------
# User's exact time array
# --------------------------------------------------

t_values = np.linspace(
    start,
    stop,
    N
)


# --------------------------------------------------
# Continuous-time convolution
#
# y(t) = integral x(tau) h(t-tau) dtau
# --------------------------------------------------

x_tau = x.subs(
    t,
    tau
)


h_shifted = h.subs(
    t,
    t - tau
)


integrand = (
    x_tau *
    h_shifted
)


f = sp.lambdify(
    (tau, t),
    integrand,
    modules=["numpy"]
)


# Integration points

tau_values = np.linspace(

    start,
    stop,

    max(
        3000,
        min(
            7000,
            N * 5
        )
    )
)


y_values = np.empty(
    N,
    dtype=float
)


# --------------------------------------------------
# Numerical integration
# --------------------------------------------------

for i, tv in enumerate(t_values):

    values = np.asarray(

        f(
            tau_values,
            tv
        ),

        dtype=float
    )


    if values.ndim == 0:

        values = np.full(
            tau_values.shape,
            float(values)
        )


    values = np.nan_to_num(
        values
    )


    if hasattr(
        np,
        "trapezoid"
    ):

        y_values[i] = np.trapezoid(
            values,
            tau_values
        )

    else:

        y_values[i] = np.trapz(
            values,
            tau_values
        )


# --------------------------------------------------
# Create Matplotlib figure
# --------------------------------------------------

fig, ax = plt.subplots(

    figsize=(10, 5.5),

    dpi=130
)


ax.plot(

    t_values,

    y_values,

    linewidth=2
)


# IMPORTANT:
# The visible graph uses the user's exact
# linspace range.

ax.set_xlim(
    start,
    stop
)


ax.set_xticks(

    np.linspace(
        start,
        stop,
        9
    )
)


ax.set_xlabel(
    "Time (t)"
)


ax.set_ylabel(
    "Amplitude"
)


ax.set_title(
    "Continuous-Time Convolution"
)


ax.grid(
    True,
    alpha=0.3
)


ax.axhline(
    0,
    linewidth=0.8
)


if start <= 0 <= stop:

    ax.axvline(
        0,
        linewidth=0.8
    )


fig.tight_layout()


# --------------------------------------------------
# Convert Matplotlib figure to PNG
# --------------------------------------------------

buffer = BytesIO()


fig.savefig(

    buffer,

    format="png",

    bbox_inches="tight"
)


plt.close(fig)


image_b64 = base64.b64encode(

    buffer.getvalue()

).decode("ascii")


# --------------------------------------------------
# Numerical values
# --------------------------------------------------

sample_step = max(
    1,
    N // 200
)


result = {

    "image": image_b64,

    "t":
        t_values[
            ::sample_step
        ].tolist(),

    "y":
        y_values[
            ::sample_step
        ].tolist(),

    "count":
        int(N)
}


result

        `);


        // --------------------------------------------------
        // Display PNG
        // --------------------------------------------------

        const imageBytes =
            Uint8Array.from(

                atob(result.image),

                c => c.charCodeAt(0)

            );


        const blob =
            new Blob(

                [imageBytes],

                {
                    type: "image/png"
                }

            );


        if (currentImage) {

            URL.revokeObjectURL(
                currentImage
            );

        }


        currentImage =
            URL.createObjectURL(blob);


        figureContainer.innerHTML = "";


        const image =
            document.createElement("img");


        image.src =
            currentImage;


        image.alt =
            "Continuous-time convolution";


        figureContainer.appendChild(
            image
        );


        downloadBtn.disabled = false;


        // --------------------------------------------------
        // Numerical output
        // --------------------------------------------------

        let text =
            `Total points: ${result.count}\\n\\n`;

        text +=
            "t\\ty(t)\\n";

        text +=
            "-------------------------\\n";


        for (
            let i = 0;
            i < result.t.length;
            i++
        ) {

            text +=
                `${Number(result.t[i]).toFixed(8)}\\t` +
                `${Number(result.y[i]).toFixed(8)}\\n`;

        }


        if (
            result.t.length <
            result.count
        ) {

            text +=
                `\\nShowing every ${
                    Math.ceil(
                        result.count /
                        result.t.length
                    )
                }th point.`;

        }


        valuesBox.textContent =
            text;


        statusText.textContent =
            "Convolution complete.";


    } catch (error) {

        console.error(error);


        statusText.textContent =
            "Error: " + error.message;


        figureContainer.innerHTML =
            '<p id="placeholder">' +
            'Could not calculate the convolution.' +
            '</p>';


        valuesBox.textContent = "";


    } finally {

        convolveBtn.disabled = false;

    }

}



// --------------------------------------------------
// Download PNG
// --------------------------------------------------

downloadBtn.addEventListener(
    "click",
    () => {

        if (!currentImage) {
            return;
        }


        const link =
            document.createElement("a");


        link.href =
            currentImage;


        link.download =
            "continuous_time_convolution.png";


        document.body.appendChild(link);


        link.click();


        link.remove();

    }
);



// --------------------------------------------------
// Convolve button
// --------------------------------------------------

convolveBtn.addEventListener(
    "click",
    convolve
);



// --------------------------------------------------
// Start Python environment
// --------------------------------------------------

loadPython();
