// Main JavaScript for Coventry University Search Engine

document.addEventListener('DOMContentLoaded', function () {
    // Smooth scroll for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });

    // Auto-focus search input on homepage
    const searchInput = document.querySelector('.search-input');
    if (searchInput && window.location.pathname === '/') {
        searchInput.focus();
    }

    // Display loading state on search button
    const searchForms = document.querySelectorAll('.search-form, .search-form-compact');
    searchForms.forEach(form => {
        form.addEventListener('submit', function (e) {
            const button = this.querySelector('button[type="submit"]');
            if (button) {
                button.innerHTML = '⏳ Searching...';
                button.disabled = true;
            }
        });
    });

    // Highlight search terms in results
    const urlParams = new URLSearchParams(window.location.search);
    const query = urlParams.get('q');
    if (query) {
        highlightSearchTerms(query);
    }
});

function highlightSearchTerms(query) {
    const terms = query.toLowerCase().split(' ').filter(t => t.length > 2);
    const resultItems = document.querySelectorAll('.result-item');

    resultItems.forEach(item => {
        const title = item.querySelector('.result-title a');
        const abstract = item.querySelector('.result-abstract');

        if (title) {
            highlightElement(title, terms);
        }
        if (abstract) {
            highlightElement(abstract, terms);
        }
    });
}

function highlightElement(element, terms) {
    let html = element.innerHTML;
    terms.forEach(term => {
        const regex = new RegExp(`(${term})`, 'gi');
        html = html.replace(regex, '<mark>$1</mark>');
    });
    element.innerHTML = html;
}

// Add mark styling
const style = document.createElement('style');
style.textContent = `
    mark {
        background-color: #fef08a;
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: 600;
    }
`;
document.head.appendChild(style);
