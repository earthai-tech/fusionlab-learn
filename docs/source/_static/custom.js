
document.addEventListener('DOMContentLoaded', function() {
    // --- CONFIGURATION ---
    // Set the date of your latest release here
    const releaseDate = new Date('2025-06-17T00:00:00Z'); // Use ISO format (YYYY-MM-DD)
    const highlightDurationDays = 7; // Highlight will disappear after this many days

    // --- LOGIC ---
    const now = new Date();
    const highlightEndDate = new Date(releaseDate);
    highlightEndDate.setDate(releaseDate.getDate() + highlightDurationDays);

    // If the current date is past the highlight period, remove the class
    if (now > highlightEndDate) {
        const sparkleElements = document.querySelectorAll('.sparkle-link');
        sparkleElements.forEach(function(element) {
            element.classList.remove('sparkle-link');
        });
        console.log('Release highlight period has ended. Sparkle effect removed.');
    } else {
        console.log('Release highlight is active.');
    }
});