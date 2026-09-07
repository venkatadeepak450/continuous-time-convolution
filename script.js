let pyodide = null;
let currentImage = null;

const convolveBtn = document.getElementById("convolveBtn");
const downloadBtn = document.getElementById("downloadBtn");
const statusText = document.getElementById("status");
const figureContainer = document.getElementById("figureContainer");
const valuesBox = document.getElementById("values");


function getInput(id) {
    return document.getElementById(id).value.trim();
}


/* =========================================================
   LOAD PYTHON / SYMPY / NUMPY / MATPLOTLIB
   ========================================================= */

async function loadPython() {

    statusText.textContent =
        "Loading Python, NumPy, SymPy and Matplotlib...";

    convolveBtn.disabled = true;

    try {

        if (typeof loadPyodide !== "function") {
            throw new Error(
                "Pyodide did not load. Check your internet connection."
            );
        }

        pyodide = await loadPyodide();

        await pyodide.loadPackage([
            "numpy",
            "sympy",
            "matplotlib"
        ]);

        statusText.textContent = "Ready.";

        convolveBtn.disabled = false;

    } catch (error) {

        console.error(error);

        statusText.textContent =
            "Could not load Python: " + error.message;
    }
}


/* =========================================================
   CONVOLUTION
   ========================================================= */

async function convolve() {

    if (!pyodide) {
        return;
    }

    const xExpr = getInput("xFunction");
    const hExpr = getInput("hFunction");
    const linspaceExpr = getInput("linspace");

    if (!xExpr || !hExpr || !linspaceExpr) {

        statusText.textContent =
            "Please fill all three inputs.";

        return;
    }

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
import re

from io import BytesIO


# =========================================================
# SYMBOLS
# =========================================================

t = sp.symbols(
    "t",
    real=True
)

tau = sp.symbols(
    "tau",
    real=True
)


# =========================================================
# ALLOWED SYMPY FUNCTIONS
# =========================================================

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

    "Heaviside": sp.Heaviside,

    "heaviside": sp.Heaviside,

    "DiracDelta": sp.DiracDelta,

    "diracdelta": sp.DiracDelta
}


# =========================================================
# FIX COMMON LOWERCASE INPUT
# =========================================================

x_input = re.sub(
    r"(?i)\\\\bsp\\\\.heaviside\\\\b",
    "sp.Heaviside",
    x_input
)

h_input = re.sub(
    r"(?i)\\\\bsp\\\\.heaviside\\\\b",
    "sp.Heaviside",
    h_input
)

x_input = re.sub(
    r"(?i)\\\\bsp\\\\.diracdelta\\\\b",
    "sp.DiracDelta",
    x_input
)

h_input = re.sub(
    r"(?i)\\\\bsp\\\\.diracdelta\\\\b",
    "sp.DiracDelta",
    h_input
)


# =========================================================
# PARSE x(t) AND h(t)
# =========================================================

try:

    x = sp.sympify(
        x_input,
        locals=local_dict
    )

    h = sp.sympify(
        h_input,
        locals=local_dict
    )

except Exception as e:

    raise ValueError(
        "Invalid SymPy expression."
    ) from e


# =========================================================
# ONLY t IS ALLOWED AS A SIGNAL VARIABLE
# =========================================================

if x.free_symbols - {t}:

    raise ValueError(
        "x(t) may contain only the variable t."
    )


if h.free_symbols - {t}:

    raise ValueError(
        "h(t) may contain only the variable t."
    )


# =========================================================
# PARSE np.linspace(start, stop, points)
# =========================================================

linspace_pattern = (
    r"^\\\\s*np\\\\.linspace\\\\("
    r"\\\\s*(.+?)\\\\s*,"
    r"\\\\s*(.+?)\\\\s*,"
    r"\\\\s*(\\\\d+)\\\\s*"
    r"\\\\)\\\\s*$"
)

match = re.fullmatch(
    linspace_pattern,
    linspace_input
)


if not match:

    raise ValueError(
        "Use exactly: "
        "np.linspace(start, stop, points)"
    )


start_text = match.group(1).strip()

stop_text = match.group(2).strip()

points_text = match.group(3).strip()


try:

    start = float(
        sp.N(
            sp.sympify(
                start_text,
                locals=local_dict
            )
        )
    )

    stop = float(
        sp.N(
            sp.sympify(
                stop_text,
                locals=local_dict
            )
        )
    )

    N = int(points_text)


except Exception as e:

    raise ValueError(
        "Invalid np.linspace values."
    ) from e


if not np.isfinite(start):

    raise ValueError(
        "Linspace start must be finite."
    )


if not np.isfinite(stop):

    raise ValueError(
        "Linspace stop must be finite."
    )


if stop <= start:

    raise ValueError(
        "Linspace stop must be greater than start."
    )


if N < 2:

    raise ValueError(
        "Linspace must contain at least 2 points."
    )


# =========================================================
# USER TIME ARRAY
# =========================================================

t_values = np.linspace(
    start,
    stop,
    N
)


# =========================================================
# DIRAC DELTA DETECTION
# =========================================================

has_delta = (
    x.has(sp.DiracDelta) or
    h.has(sp.DiracDelta)
)


# =========================================================
# SYMBOLIC CONTINUOUS-TIME CONVOLUTION
#
# y(t) = integral x(tau) h(t-tau) dtau
#
# SymPy is used first. This is especially important
# for DiracDelta because an impulse should NOT be
# sampled as an ordinary numerical function.
# =========================================================

if has_delta:

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

    try:

        symbolic_y = sp.integrate(
            integrand,
            (tau, -sp.oo, sp.oo)
        )

    except Exception:

        symbolic_y = sp.Integral(
            integrand,
            (tau, -sp.oo, sp.oo)
        )


    if isinstance(
        symbolic_y,
        sp.Integral
    ):

        # Try SymPy's convolution implementation
        try:

            symbolic_y = sp.convolution(
                x,
                h,
                t
            )

        except Exception:

            raise ValueError(
                "SymPy could not symbolically evaluate "
                "the DiracDelta convolution. "
                "Try a simpler expression."
            )


    symbolic_y = sp.simplify(
        symbolic_y
    )


    # -----------------------------------------------------
    # Numerical evaluation of symbolic result
    # -----------------------------------------------------

    try:

        y_func = sp.lambdify(
            t,
            symbolic_y,
            modules=["numpy"]
        )

        y_values = np.asarray(
            y_func(t_values),
            dtype=float
        )

    except Exception as e:

        raise ValueError(
            "The symbolic convolution could not "
            "be evaluated numerically."
        ) from e


    if y_values.ndim == 0:

        y_values = np.full(
            t_values.shape,
            float(y_values)
        )


    y_values = np.nan_to_num(
        y_values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


# =========================================================
# ORDINARY FUNCTIONS
#
# Numerical continuous-time convolution
# =========================================================

else:

    integrand = (
        x.subs(t, tau) *
        h.subs(t, t - tau)
    )


    f = sp.lambdify(
        (tau, t),
        integrand,
        modules=["numpy"]
    )


    integration_points = max(
        3000,
        min(
            7000,
            N * 5
        )
    )


    tau_values = np.linspace(
        start,
        stop,
        integration_points
    )


    y_values = np.empty(
        N,
        dtype=float
    )


    for i, tv in enumerate(t_values):

        try:

            values = np.asarray(
                f(
                    tau_values,
                    tv
                ),
                dtype=float
            )

        except Exception as e:

            raise ValueError(
                "The functions could not be "
                "evaluated numerically."
            ) from e


        if values.ndim == 0:

            values = np.full(
                tau_values.shape,
                float(values)
            )


        if values.shape != tau_values.shape:

            raise ValueError(
                "Invalid numerical function output."
            )


        values = np.nan_to_num(
            values,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
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


# =========================================================
# MATPLOTLIB FIGURE
# =========================================================

fig, ax = plt.subplots(

    figsize=(10, 5.5),

    dpi=130
)


ax.plot(
    t_values,
    y_values,
    linewidth=2
)


# EXACT USER RANGE

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


# =========================================================
# PNG
# =========================================================

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


# =========================================================
# NUMERICAL VALUES
# =========================================================

sample_step = max(
    1,
    int(np.ceil(N / 200))
)


result = {

    "image":
        image_b64,

    "t":
        t_values[
            ::sample_step
        ].tolist(),

    "y":
        y_values[
            ::sample_step
        ].tolist(),

    "count":
        int(N),

    "symbolic":
        str(
            symbolic_y
        ) if has_delta else None
}


result

        `);


        /* =================================================
           DISPLAY PNG
           ================================================= */

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


        /* =================================================
           NUMERICAL VALUES
           ================================================= */

        let text =
            "Total points: " +
            result.count +
            "\\n\\n";


        if (result.symbolic !== null) {

            text +=
                "Symbolic convolution:\\n";

            text +=
                result.symbolic +
                "\\n\\n";
        }


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
                Number(
                    result.t[i]
                ).toFixed(8) +
                "\\t" +
                Number(
                    result.y[i]
                ).toFixed(8) +
                "\\n";
        }


        if (
            result.t.length <
            result.count
        ) {

            text +=
                "\\nShowing every " +
                Math.ceil(
                    result.count /
                    result.t.length
                ) +
                "th point.";
        }


        valuesBox.textContent =
            text;


        statusText.textContent =
            result.symbolic !== null
                ? "Convolution complete (symbolic DiracDelta handling)."
                : "Convolution complete.";


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


/* =========================================================
   DOWNLOAD
   ========================================================= */

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


/* =========================================================
   BUTTON
   ========================================================= */

convolveBtn.addEventListener(
    "click",
    convolve
);


/* =========================================================
   START PYODIDE
   ========================================================= */

loadPython();
