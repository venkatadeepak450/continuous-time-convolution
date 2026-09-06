let pyodide = null;
let figureData = null;


// ==================================================
// LOAD PYTHON + NUMPY + MATPLOTLIB
// ==================================================

async function loadPython() {

    const status =
        document.getElementById("status");

    const button =
        document.getElementById("convolve-button");


    status.textContent =
        "Loading Python and Matplotlib...";


    try {

        pyodide = await loadPyodide();

        await pyodide.loadPackage([
            "numpy",
            "matplotlib"
        ]);


        status.textContent =
            "Python ready. Enter your signals.";

        button.disabled = false;


    } catch (error) {

        console.error(error);

        status.textContent =
            "Failed to load Python or Matplotlib.";

    }

}


// ==================================================
// CALCULATE CONVOLUTION
// ==================================================

async function calculateConvolution() {

    const status =
        document.getElementById("status");

    const output =
        document.getElementById("output");

    const figureContainer =
        document.getElementById("figure-container");

    const downloadButton =
        document.getElementById("download-button");


    try {

        status.textContent =
            "Calculating convolution...";


        // ==================================================
        // READ USER INPUT
        // ==================================================

        const xFunction =
            document
                .getElementById("x-function")
                .value
                .trim();


        const hFunction =
            document
                .getElementById("h-function")
                .value
                .trim();


        const linspace =
            document
                .getElementById("linspace")
                .value
                .trim();


        // ==================================================
        // CHECK INPUT
        // ==================================================

        if (!xFunction) {

            throw new Error(
                "Please enter x(t)."
            );

        }


        if (!hFunction) {

            throw new Error(
                "Please enter h(t)."
            );

        }


        if (!linspace) {

            throw new Error(
                "Please enter the common linspace."
            );

        }


        // ==================================================
        // PYTHON PROGRAM
        // ==================================================

        const pythonCode = `

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import base64

from io import BytesIO


# ==================================================
# COMMON TIME AXIS
# ==================================================

t = np.asarray(

    ${linspace},

    dtype=float

)


# ==================================================
# VALIDATE TIME AXIS
# ==================================================

if t.ndim != 1:

    raise ValueError(

        "The linspace must be one-dimensional."

    )


if len(t) < 2:

    raise ValueError(

        "The linspace needs at least 2 points."

    )


# ==================================================
# SIGNAL x(t)
# ==================================================

x = np.asarray(

    ${xFunction},

    dtype=float

)


# ==================================================
# SIGNAL h(t)
# ==================================================

h = np.asarray(

    ${hFunction},

    dtype=float

)


# ==================================================
# VALIDATE SIGNALS
# ==================================================

if x.ndim != 1:

    raise ValueError(

        "x(t) must produce a one-dimensional array."

    )


if h.ndim != 1:

    raise ValueError(

        "h(t) must produce a one-dimensional array."

    )


if len(x) != len(t):

    raise ValueError(

        "x(t) must produce exactly one value "
        "for every point in the linspace."

    )


if len(h) != len(t):

    raise ValueError(

        "h(t) must produce exactly one value "
        "for every point in the linspace."

    )


# ==================================================
# SAMPLING INTERVAL
# ==================================================

dt = float(

    abs(t[1] - t[0])

)


# ==================================================
# CHECK UNIFORM SAMPLING
# ==================================================

if not np.allclose(

    np.diff(t),

    np.diff(t)[0]

):

    raise ValueError(

        "The linspace must have uniform spacing."

    )


# ==================================================
# CONTINUOUS-TIME CONVOLUTION
# ==================================================

y = np.convolve(

    x,

    h,

    mode="full"

) * dt


# ==================================================
# CONVOLUTION TIME AXIS
# ==================================================

t_conv = np.linspace(

    t[0] + t[0],

    t[-1] + t[-1],

    len(y)

)


# ==================================================
# CREATE COMPLETE MATPLOTLIB FIGURE
# ==================================================

fig, ax = plt.subplots(

    figsize=(10, 6),

    dpi=100

)


# ==================================================
# PLOT
# ==================================================

ax.plot(

    t_conv,

    y,

    linewidth=2

)


# ==================================================
# DISPLAY RANGE
# ==================================================

ax.set_xlim(

    -10,

    10

)


# ==================================================
# TITLE
# ==================================================

ax.set_title(

    "Continuous-Time Convolution",

    fontsize=16

)


# ==================================================
# AXIS LABELS
# ==================================================

ax.set_xlabel(

    "Time (t)",

    fontsize=13

)


ax.set_ylabel(

    "Amplitude",

    fontsize=13

)


# ==================================================
# GRID
# ==================================================

ax.grid(

    True,

    linewidth=0.8,

    alpha=0.7

)


# ==================================================
# LAYOUT
# ==================================================

fig.tight_layout()


# ==================================================
# CONVERT FIGURE TO PNG
# ==================================================

buffer = BytesIO()


fig.savefig(

    buffer,

    format="png",

    dpi=100,

    facecolor="white"

)


plt.close(fig)


buffer.seek(0)


# ==================================================
# ENCODE IMAGE
# ==================================================

image_base64 = base64.b64encode(

    buffer.read()

).decode("ascii")


# ==================================================
# NUMERICAL OUTPUT
# ==================================================

output_lines = []


output_lines.append(

    "Index\\tTime\\t\\tConvolution"

)


output_lines.append(

    "---------------------------------------------"

)


for i in range(len(y)):

    output_lines.append(

        f"{i + 1}\\t"
        f"{t_conv[i]:.6f}\\t"
        f"{y[i]:.10f}"

    )


numerical_output = "\\n".join(

    output_lines

)

`;


        // ==================================================
        // RUN PYTHON
        // ==================================================

        await pyodide.runPythonAsync(

            pythonCode

        );


        // ==================================================
        // GET RESULTS
        // ==================================================

        const imageValue =
            pyodide.globals.get(
                "image_base64"
            );


        const numericalValue =
            pyodide.globals.get(
                "numerical_output"
            );


        if (
            imageValue === undefined ||
            imageValue === null
        ) {

            throw new Error(

                "Matplotlib did not produce an image."

            );

        }


        // ==================================================
        // CONVERT RESULTS
        // ==================================================

        const imageBase64 =
            String(imageValue);


        const numericalOutput =
            String(numericalValue);


        // ==================================================
        // DISPLAY FIGURE
        // ==================================================

        figureContainer.innerHTML = "";


        const image =
            document.createElement("img");


        image.src =
            "data:image/png;base64," +
            imageBase64;


        image.alt =
            "Continuous-Time Convolution";


        figureContainer.appendChild(

            image

        );


        // ==================================================
        // DISPLAY NUMERICAL OUTPUT
        // ==================================================

        output.textContent =
            numericalOutput;


        // ==================================================
        // ENABLE DOWNLOAD
        // ==================================================

        figureData =
            "data:image/png;base64," +
            imageBase64;


        downloadButton.disabled = false;


        // ==================================================
        // SUCCESS
        // ==================================================

        status.textContent =
            "Convolution completed successfully.";

    }


    catch (error) {

        console.error(error);


        status.textContent =
            "Error while calculating.";


        output.textContent =
            error.toString();

    }

}


// ==================================================
// DOWNLOAD FIGURE
// ==================================================

document
    .getElementById("download-button")
    .addEventListener(

        "click",

        function () {

            if (!figureData) {

                return;

            }


            const link =
                document.createElement("a");


            link.href =
                figureData;


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

    );


// ==================================================
// CONVOLVE BUTTON
// ==================================================

document
    .getElementById("convolve-button")
    .addEventListener(

        "click",

        calculateConvolution

    );


// ==================================================
// START PYTHON
// ==================================================

loadPython();
