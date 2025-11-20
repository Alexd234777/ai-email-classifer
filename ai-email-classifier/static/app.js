document.addEventListener("DOMContentLoaded", () => {
    const html = document.documentElement;
    const toggle = document.getElementById("themeToggle");

    let theme = localStorage.getItem("theme") || "light";
    html.dataset.theme = theme;

    toggle.addEventListener("click", () => {
        theme = theme === "light" ? "dark" : "light";
        html.dataset.theme = theme;
        localStorage.setItem("theme", theme);
    });
});


