document.addEventListener("DOMContentLoaded", () => {
  const form = document.querySelector("form");
  const resultDiv = document.getElementById("result");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());

    // Convert range/number inputs from string to float
    for (let key in data) {
      if (!isNaN(data[key])) {
        data[key] = parseFloat(data[key]);
      }
    }

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
      });

      const result = await response.json();

      if (result && result.probability !== undefined) {
        const percent = (result.probability * 100).toFixed(2);
        resultDiv.textContent = `Cancellation Probability: ${percent}%`;
      } else {
        resultDiv.textContent = "Prediction failed.";
      }
    } catch (error) {
      console.error("Error:", error);
      resultDiv.textContent = "Error during prediction.";
    }
  });
});
