let pyodide = null;
let generatedImage = null;

const convolveButton = document.getElementById("convolve-button");
const downloadButton = document.getElementById("download-button");
const status = document.getElementById("status");
const output = document.getElementById("output");
const figureContainer = document.getElementById("figure-container");


// ============================================================
// LOAD PYTHON
// ============================================================

async function loadPython() {

    try {

        status.textContent =
            "Loading Python and Matplotlib...";

        pyodide = await loadPyodide();

        status.textContent =
            "Loading NumPy and Matplotlib...";

        await pyodide.loadPackage([
            "numpy",
            "matplotlib"
        ]);

        status.textContent = "Ready.";

        convolveButton.disabled = false;

    } catch (error) {

        console.error(error);

        status.textContent =
            "Failed to load Python and Matplotlib.";

        convolveButton.disabled = true;
    }
}


// ============================================================
// PERFORM CONVOLUTION
// ============================================================

async function performConvolution() {

    if (!pyodide) {

        status.textContent =
            "Python is still loading...";

        return;
    }


    const xFunction =
        document.getElementById("x-function").value.trim();

    const hFunction =
        document.getElementById("h-function").value.trim();

    const linspaceExpression =
        document.getElementById("linspace").value.trim();


    // --------------------------------------------------------
    // CHECK INPUT
    // --------------------------------------------------------

    if (
        xFunction === "" ||
        hFunction === "" ||
        linspaceExpression === ""
    ) {

        status.textContent =
            "Please fill in all three fields.";

        return;
    }


    convolveButton.disabled = true;
    downloadButton.disabled = true;

    status.textContent =
        "Calculating convolution...";

    output.textContent = "";

    generatedImage = null;

    figureContainer.innerHTML =
        "<p id='placeholder'>Generating Matplotlib figure...</p>";


    try {

        // ----------------------------------------------------
        // SEND EXPRESSIONS TO PYTHON
        // ----------------------------------------------------

        pyodide.globals.set(
            "x_expression",
            xFunction
        );

        pyodide.globals.set(
            "h_expression",
            hFunction
        );

        pyodide.globals.set(
            "t_expression",
            linspaceExpression
        );


        // ----------------------------------------------------
        // PYTHON CODE
        // ----------------------------------------------------

        const result =
            await pyodide.runPythonAsync(`

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import base64
from io import BytesIO


# ==========================================================
# 1. CREATE THE USER'S COMMON TIME ARRAY
# ==========================================================

t = np.asarray(
    eval(
        t_expression,
        {
            "np": np
        }
    ),
    dtype=float
)


# ==========================================================
# 2. VALIDATE TIME ARRAY
# ==========================================================

if t.ndim != 1:

    raise ValueError(
        "Common linspace must produce a 1-D array."
    )


if len(t) < 2:

    raise ValueError(
        "Common linspace must contain at least 2 points."
    )


if not np.all(np.isfinite(t)):

    raise ValueError(
        "Common linspace contains NaN or infinity."
    )


# ==========================================================
# 3. CHECK THAT t IS STRICTLY INCREASING
# ==========================================================

dt_values = np.diff(t)


if np.any(dt_values <= 0):

    raise ValueError(
        "Common linspace must be strictly increasing."
    )


# ==========================================================
# 4. CHECK UNIFORM SPACING
# ==========================================================

dt = float(dt_values[0])


if not np.allclose(
    dt_values,
    dt,
    rtol=1e-8,
    atol=1e-12
):

    raise ValueError(
        "Common linspace must have uniform spacing."
    )


# ==========================================================
# 5. EVALUATE x(t)
# ==========================================================

x = np.asarray(
    eval(
        x_expression,
        {
            "np": np,
            "t": t
        }
    ),
    dtype=float
)


# ==========================================================
# 6. EVALUATE h(t)
# ==========================================================

h = np.asarray(
    eval(
        h_expression,
        {
            "np": np,
            "t": t
        }
    ),
    dtype=float
)


# ==========================================================
# 7. HANDLE SCALAR OUTPUTS
# ==========================================================

if x.ndim == 0:

    x = np.full(
        t.shape,
        float(x)
    )


if h.ndim == 0:

    h = np.full(
        t.shape,
        float(h)
    )


# ==========================================================
# 8. VALIDATE SIGNAL DIMENSIONS
# ==========================================================

if x.ndim != 1:

    raise ValueError(
        "x(t) must produce a 1-D array."
    )


if h.ndim != 1:

    raise ValueError(
        "h(t) must produce a 1-D array."
    )


# ==========================================================
# 9. VALIDATE SIGNAL LENGTHS
# ==========================================================

if len(x) != len(t):

    raise ValueError(
        "x(t) must produce exactly one value for every t value."
    )


if len(h) != len(t):

    raise ValueError(
        "h(t) must produce exactly one value for every t value."
    )


# ==========================================================
# 10. CHECK SIGNAL VALUES
# ==========================================================

if not np.all(np.isfinite(x)):

    raise ValueError(
        "x(t) contains NaN or infinity."
    )


if not np.all(np.isfinite(h)):

    raise ValueError(
        "h(t) contains NaN or infinity."
    )


# ==========================================================
# 11. CONTINUOUS-TIME CONVOLUTION
#
# y(t) = integral x(tau) h(t-tau) d(tau)
#
# Numerical approximation:
#
# y[n] ≈ sum(x[k] h[n-k]) dt
# ==========================================================

y_full = (
    np.convolve(
        x,
        h,
        mode="full"
    )
    * abs(dt)
)


# ==========================================================
# 12. CREATE THE ACTUAL FULL-CONVOLUTION TIME GRID
#
# If the input time array is:
#
#     t[0] ... t[-1]
#
# then the full convolution time range is:
#
#     t[0] + t[0]
#
# to:
#
#     t[-1] + t[-1]
#
# This grid is ONLY used internally.
# ==========================================================

full_start = float(t[0] + t[0])

full_time = (
    full_start
    +
    np.arange(len(y_full)) * dt
)


# ==========================================================
# 13. MAP THE FULL CONVOLUTION ONTO THE USER'S t GRID
#
# THIS IS THE IMPORTANT FIX.
#
# The FINAL output is evaluated/displayed at exactly:
#
#     t
#
# supplied by the user.
#
# Therefore the graph can NEVER accidentally use
# the -40 -> +40 full-convolution range.
# ==========================================================

y = np.interp(
    t,
    full_time,
    y_full,
    left=0.0,
    right=0.0
)


# ==========================================================
# 14. FINAL SIZE CHECK
# ==========================================================

if len(y) != len(t):

    raise ValueError(
        "Convolution output does not match the common t array."
    )


# ==========================================================
# 15. CREATE MATPLOTLIB FIGURE
# ==========================================================

fig, ax = plt.subplots(
    figsize=(10, 6),
    dpi=150
)


# ==========================================================
# 16. PLOT USING THE USER'S t ARRAY
#
# x-axis = EXACTLY t
# y-axis = convolution evaluated at t
# ==========================================================

ax.plot(
    t,
    y,
    linewidth=2
)


# ==========================================================
# 17. FORCE EXACT X RANGE
# ==========================================================

ax.set_xlim(
    float(t[0]),
    float(t[-1])
)


# ==========================================================
# 18. UNIFORMLY SPACED X TICKS
# ==========================================================

number_of_ticks = 9

x_ticks = np.linspace(
    float(t[0]),
    float(t[-1]),
    number_of_ticks
)

ax.set_xticks(x_ticks)


# ==========================================================
# 19. LABELS
# ==========================================================

ax.set_xlabel(
    "Time (t)"
)

ax.set_ylabel(
    "Amplitude"
)

ax.set_title(
    "Continuous-Time Convolution"
)


# ==========================================================
# 20. GRID
# ==========================================================

ax.grid(
    True,
    alpha=0.3
)


# ==========================================================
# ZERO AXES
# ==========================================================

ax.axhline(
    0,
    linewidth=0.8
)


if float(t[0]) <= 0 <= float(t[-1]):

    ax.axvline(
        0,
        linewidth=0.8
    )


# ==========================================================
# 21. FINALIZE FIGURE
# ==========================================================

fig.tight_layout()


# ==========================================================
# 22. CONVERT FIGURE TO PNG
# ==========================================================

buffer = BytesIO()

fig.savefig(
    buffer,
    format="png",
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)

buffer.seek(0)


image_base64 = base64.b64encode(
    buffer.read()
).decode("utf-8")


# ==========================================================
# 23. NUMERICAL OUTPUT
# ==========================================================

numerical_output = ""

for i in range(len(t)):

    numerical_output += (
        f"{i + 1:4d}    "
        f"t = {t[i]: .10f}    "
        f"y = {y[i]: .10f}\\\\n"
    )


# ==========================================================
# 24. RETURN IMAGE + NUMERICAL DATA
# ==========================================================

image_base64 + "\\n---NUMERICAL_DATA---\\n" + numerical_output

            `);


        // ====================================================
        // SPLIT IMAGE AND DATA
        // ====================================================

        const separator =
            "\n---NUMERICAL_DATA---\n";

        const separatorIndex =
            result.indexOf(separator);


        if (separatorIndex === -1) {

            throw new Error(
                "Invalid result returned from Python."
            );
        }


        const imageBase64 =
            result.substring(
                0,
                separatorIndex
            );

        const numericalData =
            result.substring(
                separatorIndex + separator.length
            );


        // ====================================================
        // DISPLAY FIGURE
        // ====================================================

        generatedImage =
            "data:image/png;base64," +
            imageBase64;


        figureContainer.innerHTML = "";


        const image =
            document.createElement("img");


        image.src =
            generatedImage;


        image.alt =
            "Continuous-Time Convolution";


        figureContainer.appendChild(
            image
        );


        // ====================================================
        // DISPLAY NUMERICAL DATA
        // ====================================================

        output.textContent =
            numericalData;


        // ====================================================
        // ENABLE DOWNLOAD
        // ====================================================

        downloadButton.disabled = false;


        status.textContent =
            "Convolution completed successfully.";

    }


    catch (error) {

        console.error(error);

        figureContainer.innerHTML =
            "<p id='placeholder'>Error while calculating convolution.</p>";

        output.textContent = "";

        status.textContent =
            "Error: " + error.message;
    }


    finally {

        convolveButton.disabled = false;
    }
}


// ============================================================
// DOWNLOAD FIGURE
// ============================================================

function downloadFigure() {

    if (!generatedImage) {
        return;
    }


    const link =
        document.createElement("a");


    link.href =
        generatedImage;


    link.download =
        "continuous_time_convolution.png";


    document.body.appendChild(
        link
    );


    link.click();


    document.body.removeChild(
        link
    );
}


// ============================================================
// BUTTON EVENTS
// ============================================================

convolveButton.addEventListener(
    "click",
    performConvolution
);


downloadButton.addEventListener(
    "click",
    downloadFigure
);


// ============================================================
// START PYTHON
// ============================================================

loadPython();
