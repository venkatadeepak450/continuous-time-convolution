let pyodide = null;
let currentImage = null;

const convolveBtn = document.getElementById("convolveBtn");
const downloadBtn = document.getElementById("downloadBtn");
const statusText = document.getElementById("status");
const figureContainer = document.getElementById("figureContainer");
const valuesBox = document.getElementById("values");


function getInput(id) {
    const el = document.getElementById(id);
    return el ? el.value.trim() : "";
}


function setStatus(text) {
    statusText.textContent = text;
}


/* =========================================================
   LOAD PYTHON / NUMPY / SYMPY / MATPLOTLIB
   ========================================================= */

async function loadPython() {

    convolveBtn.disabled = true;

    setStatus(
        "Loading Python, NumPy, SymPy and Matplotlib..."
    );

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

        convolveBtn.disabled = false;

        setStatus("Ready.");

    } catch (error) {

        console.error(error);

        setStatus(
            "Could not load Python: " +
            error.message
        );
    }
}


/* =========================================================
   CONVOLUTION
   ========================================================= */

async function convolve() {

    if (!pyodide) {

        setStatus(
            "Python is still loading."
        );

        return;
    }


    const xExpr =
        getInput("xFunction");

    const hExpr =
        getInput("hFunction");

    const linspaceExpr =
        getInput("linspace");


    if (
        !xExpr ||
        !hExpr ||
        !linspaceExpr
    ) {

        setStatus(
            "Please fill all three inputs."
        );

        return;
    }


    convolveBtn.disabled = true;

    downloadBtn.disabled = true;

    setStatus(
        "Calculating convolution..."
    );


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
# SYMPY FUNCTIONS
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
# NORMALIZE COMMON INPUT SPELLINGS
# =========================================================

def normalize(text):

    text = re.sub(
        r"(?i)\\bsp\\.heaviside\\b",
        "sp.Heaviside",
        text
    )

    text = re.sub(
        r"(?i)\\bsp\\.diracdelta\\b",
        "sp.DiracDelta",
        text
    )

    return text


x_input_clean = normalize(
    x_input
)

h_input_clean = normalize(
    h_input
)


# =========================================================
# PARSE SIGNALS
# =========================================================

try:

    x = sp.sympify(
        x_input_clean,
        locals=local_dict
    )

    h = sp.sympify(
        h_input_clean,
        locals=local_dict
    )

except Exception as e:

    raise ValueError(
        "Invalid SymPy expression. "
        "Examples: sp.sin(t), "
        "sp.exp(-t**2), "
        "sp.Heaviside(t), "
        "sp.DiracDelta(t)."
    ) from e


# =========================================================
# CHECK VARIABLES
# =========================================================

if x.free_symbols - {t}:

    raise ValueError(
        "x(t) may contain only t."
    )


if h.free_symbols - {t}:

    raise ValueError(
        "h(t) may contain only t."
    )


# =========================================================
# PARSE LINSPACE
# =========================================================

pattern = (

    r"^\\s*np\\.linspace\\("

    r"\\s*(.+?)\\s*,"

    r"\\s*(.+?)\\s*,"

    r"\\s*(\\d+)\\s*"

    r"\\)\\s*$"

)


match = re.fullmatch(
    pattern,
    linspace_input
)


if not match:

    raise ValueError(
        "Use np.linspace(start, stop, points)."
    )


start = float(
    sp.N(
        sp.sympify(
            match.group(1),
            locals=local_dict
        )
    )
)


stop = float(
    sp.N(
        sp.sympify(
            match.group(2),
            locals=local_dict
        )
    )
)


N = int(
    match.group(3)
)


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
# NUMERICAL EVALUATION HELPER
# =========================================================

def evaluate_expression(expr):

    fn = sp.lambdify(
        t,
        expr,
        modules=["numpy"]
    )

    values = np.asarray(
        fn(t_values),
        dtype=float
    )


    if values.ndim == 0:

        values = np.full(
            t_values.shape,
            float(values)
        )


    if values.shape != t_values.shape:

        raise ValueError(
            "Expression did not produce "
            "one value per t point."
        )


    return np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


# =========================================================
# DIRAC DELTA INFORMATION
# =========================================================

def delta_info(expr):

    deltas = list(
        expr.atoms(
            sp.DiracDelta
        )
    )


    if len(deltas) != 1:

        return None


    delta = deltas[0]

    argument = delta.args[0]

    derivative = sp.simplify(
        sp.diff(
            argument,
            t
        )
    )


    if derivative == 0:

        return None


    if not derivative.is_number:

        return None


    roots = sp.solve(
        sp.Eq(
            argument,
            0
        ),
        t
    )


    if len(roots) != 1:

        return None


    location = sp.simplify(
        roots[0]
    )


    scale = sp.simplify(
        1 /
        sp.Abs(
            derivative
        )
    )


    return (
        delta,
        location,
        scale
    )


# =========================================================
# SIMPLE DIRAC DELTA CONVOLUTION
# =========================================================

def delta_convolution(
    x_expr,
    h_expr
):

    x_deltas = list(
        x_expr.atoms(
            sp.DiracDelta
        )
    )

    h_deltas = list(
        h_expr.atoms(
            sp.DiracDelta
        )
    )


    # Both containing delta:
    # handle separately below only if possible.

    if x_deltas and h_deltas:

        if (
            len(x_deltas) == 1 and
            len(h_deltas) == 1
        ):

            xi = delta_info(
                x_expr
            )

            hi = delta_info(
                h_expr
            )

            if xi is not None and hi is not None:

                xd, a, xs = xi

                hd, b, hs = hi

                return sp.simplify(
                    xs * hs *
                    sp.DiracDelta(
                        t - a - b
                    )
                )

        return None


    # x(t) contains delta

    if len(x_deltas) == 1:

        info = delta_info(
            x_expr
        )

        if info is None:

            return None


        delta, a, scale = info


        ordinary = sp.simplify(
            x_expr / delta
        )


        result = (
            scale *
            ordinary.subs(t, a) *
            h_expr.subs(
                t,
                t - a
            )
        )


        return sp.simplify(
            result
        )


    # h(t) contains delta

    if len(h_deltas) == 1:

        info = delta_info(
            h_expr
        )

        if info is None:

            return None


        delta, b, scale = info


        ordinary = sp.simplify(
            h_expr / delta
        )


        result = (
            scale *
            ordinary.subs(t, b) *
            x_expr.subs(
                t,
                t - b
            )
        )


        return sp.simplify(
            result
        )


    return None


# =========================================================
# CONVOLUTION
# =========================================================

has_delta = (

    x.has(
        sp.DiracDelta
    )

    or

    h.has(
        sp.DiracDelta
    )

)


symbolic_y = None


# =========================================================
# DIRAC DELTA PATH
# =========================================================

if has_delta:

    symbolic_y = delta_convolution(
        x,
        h
    )


    # Try general symbolic integration if
    # the simple delta handler did not work.

    if symbolic_y is None:

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

            candidate = sp.integrate(
                integrand,
                (
                    tau,
                    -sp.oo,
                    sp.oo
                )
            )

        except Exception:

            candidate = None


        if (
            candidate is not None
            and
            not isinstance(
                candidate,
                sp.Integral
            )
        ):

            symbolic_y = sp.simplify(
                candidate
            )


    if (
        symbolic_y is None
        or
        isinstance(
            symbolic_y,
            sp.Integral
        )
    ):

        raise ValueError(
            "This DiracDelta form could not "
            "be evaluated automatically. "
            "Use a linear impulse such as "
            "sp.DiracDelta(t-a)."
        )


    y_values = evaluate_expression(
        symbolic_y
    )


# =========================================================
# ORDINARY NUMERICAL CONVOLUTION
# =========================================================

else:

    integrand = (

        x.subs(
            t,
            tau
        )

        *

        h.subs(
            t,
            t - tau
        )

    )


    fn = sp.lambdify(
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


    for i, tv in enumerate(
        t_values
    ):

        values = np.asarray(

            fn(
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


# EXACT USER REQUESTED RANGE

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
# NUMERICAL OUTPUT
# =========================================================

sample_step = max(

    1,

    int(
        np.ceil(
            N / 200
        )
    )

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
        (
            str(symbolic_y)
            if symbolic_y is not None
            else None
        )

}


result
        `);


        /* =================================================
           DISPLAY IMAGE
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
            URL.createObjectURL(
                blob
            );


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


        if (
            result.symbolic !== null
        ) {

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


        if (
            result.symbolic !== null
        ) {

            setStatus(
                "Convolution complete — " +
                "DiracDelta handled symbolically."
            );

        } else {

            setStatus(
                "Convolution complete."
            );
        }


    } catch (error) {

        console.error(error);


        setStatus(
            "Error: " +
            (error.message || error)
        );


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
   DOWNLOAD PNG
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


        document.body.appendChild(
            link
        );


        link.click();


        link.remove();

    }
);


/* =========================================================
   CONVOLVE BUTTON
   ========================================================= */

convolveBtn.addEventListener(
    "click",
    convolve
);


/* =========================================================
   START PYODIDE
   ========================================================= */

loadPython();
