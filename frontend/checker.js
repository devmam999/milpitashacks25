fetch("/upload", {
  method: "POST",
  body: formData
})
  .then(response => {
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    return response.json();
  })
  .then(data => {
    console.log("API Response:", data); // Debug: Log the full response
    if (!data || typeof data !== "object") {
      console.error("Invalid response format:", data);
      return;
    }
    if (data.success) {
      if (Array.isArray(data.faces) && data.faces.length > 0) {
        console.log("Faces detected:", data.faces);
      } else {
        console.log("Emotion:", data.emotion || "N/A");
        console.log("Score:", data.score || 0);
        console.log("Distress:", data.distress || "N/A");
      }
    } else {
      console.error("Detection error:", data.error || "Unknown error");
    }
  })
  .catch(error => {
    console.error("Fetch error:", error.message);
  });