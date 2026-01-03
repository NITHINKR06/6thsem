const fs = require('fs');
const path = require('path');

const outDir = path.join(__dirname, 'public/diagrams');

const diagrams = {
    'bio_vs_artificial_neuron.svg': `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 400">
  <rect width="800" height="400" fill="#fff" />
  <text x="200" y="30" font-family="Arial" font-weight="bold" text-anchor="middle">Biological Neuron</text>
  <text x="600" y="30" font-family="Arial" font-weight="bold" text-anchor="middle">Artificial Neuron</text>
  
  <g transform="translate(50, 60)">
    <circle cx="150" cy="150" r="40" fill="#ffcc80" stroke="#f57c00" stroke-width="2"/>
    <text x="150" y="155" text-anchor="middle">Soma</text>
    <path d="M 110 150 L 50 100" stroke="#f57c00" stroke-width="2"/>
    <path d="M 110 150 L 50 200" stroke="#f57c00" stroke-width="2"/>
    <text x="30" y="100">Dendrites</text>
    <path d="M 190 150 L 290 150" stroke="#f57c00" stroke-width="4"/>
    <text x="240" y="140">Axon</text>
  </g>

  <g transform="translate(450, 60)">
    <circle cx="150" cy="150" r="40" fill="#e1bee7" stroke="#7b1fa2" stroke-width="2"/>
    <text x="150" y="145" text-anchor="middle" font-size="12">Summation</text>
    <text x="150" y="165" text-anchor="middle" font-size="12">Activation</text>
    <path d="M 50 100 L 115 135" stroke="#7b1fa2" stroke-width="2" marker-end="url(#arrow)"/>
    <path d="M 50 200 L 115 165" stroke="#7b1fa2" stroke-width="2" marker-end="url(#arrow)"/>
    <text x="40" y="100">x1 (w1)</text>
    <text x="40" y="200">x2 (w2)</text>
    <path d="M 190 150 L 250 150" stroke="#7b1fa2" stroke-width="2" marker-end="url(#arrow)"/>
    <text x="270" y="155">Output (y)</text>
  </g>
  <defs><marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#7b1fa2" /></marker></defs>
</svg>`,

    'backpropagation.svg': `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 300">
  <rect width="800" height="300" fill="#f0f4c3" />
  <text x="400" y="30" font-family="Arial" font-weight="bold" text-anchor="middle">Backpropagation in Neural Network</text>
  
  <g transform="translate(100, 100)">
    <circle cx="0" cy="50" r="20" fill="#fff" stroke="#333"/>
    <circle cx="0" cy="150" r="20" fill="#fff" stroke="#333"/>
    <text x="0" y="200" text-anchor="middle">Input</text>

    <circle cx="200" cy="50" r="20" fill="#fff" stroke="#333"/>
    <circle cx="200" cy="150" r="20" fill="#fff" stroke="#333"/>
    <text x="200" y="200" text-anchor="middle">Hidden</text>

    <circle cx="400" cy="100" r="20" fill="#fff" stroke="#333"/>
    <text x="400" y="200" text-anchor="middle">Output</text>
  </g>

  <!-- Forward Pass -->
  <path d="M 120 150 L 280 150" stroke="green" stroke-width="3" marker-end="url(#arrowGreen)"/>
  <text x="200" y="250" fill="green" font-weight="bold">Forward Pass (Compute Output)</text>

  <!-- Backward Pass -->
  <path d="M 480 80 C 350 20, 250 20, 120 80" stroke="red" stroke-width="3" stroke-dasharray="5,5" fill="none" marker-end="url(#arrowRed)"/>
  <text x="300" y="60" fill="red" font-weight="bold">Backward Pass (Error Update)</text>

  <defs>
    <marker id="arrowGreen" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="green" /></marker>
    <marker id="arrowRed" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="red" /></marker>
  </defs>
</svg>`,

    'naive_bayes.svg': `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 400">
  <rect width="600" height="400" fill="#e0f7fa" />
  <text x="300" y="40" font-family="Arial" font-weight="bold" text-anchor="middle">Naive Bayes Structure</text>
  
  <rect x="250" y="80" width="100" height="50" rx="10" fill="#00acc1" stroke="#006064"/>
  <text x="300" y="110" fill="white" text-anchor="middle" font-weight="bold">Class (C)</text>

  <g transform="translate(50, 250)">
    <circle cx="50" cy="0" r="30" fill="#b2ebf2" stroke="#006064"/> <text x="50" y="5" text-anchor="middle">Feature 1</text>
    <circle cx="180" cy="0" r="30" fill="#b2ebf2" stroke="#006064"/> <text x="180" y="5" text-anchor="middle">Feature 2</text>
    <circle cx="310" cy="0" r="30" fill="#b2ebf2" stroke="#006064"/> <text x="310" y="5" text-anchor="middle">Feature 3</text>
    <circle cx="440" cy="0" r="30" fill="#b2ebf2" stroke="#006064"/> <text x="440" y="5" text-anchor="middle">Feature N</text>
  </g>

  <line x1="300" y1="130" x2="100" y2="220" stroke="#00838f" stroke-width="2"/>
  <line x1="300" y1="130" x2="230" y2="220" stroke="#00838f" stroke-width="2"/>
  <line x1="300" y1="130" x2="360" y2="220" stroke="#00838f" stroke-width="2"/>
  <line x1="300" y1="130" x2="490" y2="220" stroke="#00838f" stroke-width="2"/>
  
  <text x="300" y="350" text-anchor="middle" font-style="italic">Features are conditionally independent given Class</text>
</svg>`,

    'locally_weighted_regression.svg': `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 400">
  <rect width="600" height="400" fill="#fff3e0" />
  <text x="300" y="30" font-family="Arial" font-weight="bold" text-anchor="middle">Locally Weighted Regression</text>
  
  <!-- Axes -->
  <line x1="50" y1="350" x2="550" y2="350" stroke="black" stroke-width="2"/>
  <line x1="50" y1="350" x2="50" y2="50" stroke="black" stroke-width="2"/>

  <!-- Points -->
  <circle cx="100" cy="300" r="3" fill="#333"/>
  <circle cx="150" cy="280" r="3" fill="#333"/>
  <circle cx="200" cy="200" r="3" fill="#333"/>
  <circle cx="250" cy="150" r="3" fill="#333"/> <!-- Target Area -->
  <circle cx="300" cy="130" r="3" fill="#333"/>
  <circle cx="350" cy="180" r="3" fill="#333"/>
  
  <!-- Query Point -->
  <line x1="250" y1="350" x2="250" y2="50" stroke="blue" stroke-dasharray="4"/>
  <text x="250" y="370" fill="blue" text-anchor="middle">Query x</text>
  
  <!-- Bell Curve for Weights -->
  <path d="M 100 340 Q 250 150 400 340" fill="none" stroke="red" stroke-width="2" opacity="0.5"/>
  <text x="420" y="330" fill="red">Weight w(i)</text>
  
  <!-- Local Fit Line -->
  <line x1="200" y1="180" x2="300" y2="120" stroke="green" stroke-width="4"/>
  <text x="320" y="100" fill="green">Local Linear Fit</text>
</svg>`,

    'inductive_learning.svg': `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 300">
  <rect width="800" height="300" fill="#f3e5f5" />
  <text x="400" y="30" font-family="Arial" font-weight="bold" text-anchor="middle">Combined Inductive Learning</text>
  
  <rect x="50" y="100" width="120" height="60" rx="5" fill="#ce93d8" stroke="#4a148c"/>
  <text x="110" y="135" text-anchor="middle" font-weight="bold">Domain Theory</text>
  
  <path d="M 170 130 L 220 130" stroke="#4a148c" stroke-width="2" marker-end="url(#arrowP)"/>
  
  <rect x="220" y="100" width="120" height="60" rx="5" fill="#e1bee7" stroke="#4a148c"/>
  <text x="280" y="135" text-anchor="middle">Initial Hypothesis</text>
  
  <path d="M 340 130 L 390 130" stroke="#4a148c" stroke-width="2" marker-end="url(#arrowP)"/>
  
  <circle cx="420" cy="130" r="30" fill="#fff" stroke="#4a148c"/>
  <text x="420" y="135" text-anchor="middle" font-size="10">Refine</text>

  <rect x="360" y="200" width="120" height="40" rx="5" fill="#fff" stroke="#333"/>
  <text x="420" y="225" text-anchor="middle">Training Examples</text>
  <path d="M 420 200 L 420 160" stroke="#333" stroke-width="2" marker-end="url(#arrowP)"/>
  
  <path d="M 450 130 L 500 130" stroke="#4a148c" stroke-width="2" marker-end="url(#arrowP)"/>

  <rect x="500" y="100" width="120" height="60" rx="5" fill="#8e24aa" stroke="#fff"/>
  <text x="560" y="135" text-anchor="middle" fill="white" font-weight="bold">Final Hypothesis</text>
  
  <defs><marker id="arrowP" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#4a148c" /></marker></defs>
</svg>`,

    'kbann.svg': `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 400">
  <rect width="600" height="400" fill="#fff8e1" />
  <text x="300" y="30" font-family="Arial" font-weight="bold" text-anchor="middle">KBANN (Knowledge Based ANN)</text>
  
  <!-- Rules -->
  <rect x="50" y="100" width="150" height="200" fill="#fff" stroke="#ff6f00" stroke-dasharray="5,5"/>
  <text x="125" y="130" text-anchor="middle" fill="#ff6f00">Rules (IF-THEN)</text>
  <text x="125" y="160" text-anchor="middle" font-size="10">IF A & B THEN C</text>
  <text x="125" y="180" text-anchor="middle" font-size="10">IF C OR D THEN E</text>
  
  <path d="M 200 200 L 250 200" stroke="#ff6f00" stroke-width="4" marker-end="url(#arrowO)"/>
  
  <!-- Neural Network -->
  <g transform="translate(300, 80)">
    <circle cx="50" cy="50" r="15" fill="#ffe0b2" stroke="#e65100"/>
    <circle cx="50" cy="150" r="15" fill="#ffe0b2" stroke="#e65100"/>
    <circle cx="150" cy="100" r="15" fill="#ffcc80" stroke="#e65100"/>
    <circle cx="250" cy="100" r="15" fill="#ffb74d" stroke="#e65100"/>
    
    <line x1="65" y1="50" x2="135" y2="100" stroke="#e65100" stroke-width="2"/>
    <line x1="65" y1="150" x2="135" y2="100" stroke="#e65100" stroke-width="2"/>
    <line x1="165" y1="100" x2="235" y2="100" stroke="#e65100" stroke-width="2"/>
    
    <text x="150" y="200" text-anchor="middle" fill="#e65100" font-weight="bold">Network Initialized from Rules</text>
  </g>
  <defs><marker id="arrowO" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#e65100" /></marker></defs>
</svg>`,

    'defense_in_depth.svg': `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500">
  <rect width="500" height="500" fill="#eceff1" />
  <text x="250" y="30" font-family="Arial" font-weight="bold" text-anchor="middle">Defense in Depth</text>
  
  <circle cx="250" cy="250" r="200" fill="#cfd8dc" stroke="#455a64"/> <text x="250" y="70" text-anchor="middle">Physical Security</text>
  <circle cx="250" cy="250" r="160" fill="#b0bec5" stroke="#455a64"/> <text x="250" y="110" text-anchor="middle">Network (Firewall)</text>
  <circle cx="250" cy="250" r="120" fill="#90a4ae" stroke="#455a64"/> <text x="250" y="150" text-anchor="middle">Host (Antivirus)</text>
  <circle cx="250" cy="250" r="80" fill="#78909c" stroke="#455a64"/> <text x="250" y="190" text-anchor="middle">Application</text>
  <circle cx="250" cy="250" r="40" fill="#546e7a" stroke="#455a64"/> 
  <text x="250" y="255" text-anchor="middle" fill="white" font-weight="bold">DATA</text>
</svg>`,

    'firewall_generations.svg': `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 300">
  <rect width="800" height="300" fill="#ffebee" />
  <text x="400" y="40" font-family="Arial" font-weight="bold" text-anchor="middle">Firewall Generations</text>
  
  <line x1="50" y1="150" x2="750" y2="150" stroke="#d32f2f" stroke-width="4" marker-end="url(#arrowR)"/>
  
  <g transform="translate(100, 100)">
    <circle cx="0" cy="50" r="10" fill="#d32f2f"/>
    <text x="0" y="80" text-anchor="middle" font-weight="bold">Gen 1</text>
    <text x="0" y="100" text-anchor="middle" font-size="12">Packet Filters</text>
  </g>
  <g transform="translate(300, 100)">
    <circle cx="0" cy="50" r="10" fill="#d32f2f"/>
    <text x="0" y="80" text-anchor="middle" font-weight="bold">Gen 2</text>
    <text x="0" y="100" text-anchor="middle" font-size="12">Stateful</text>
  </g>
  <g transform="translate(500, 100)">
    <circle cx="0" cy="50" r="10" fill="#d32f2f"/>
    <text x="0" y="80" text-anchor="middle" font-weight="bold">Gen 3</text>
    <text x="0" y="100" text-anchor="middle" font-size="12">Application/Proxy</text>
  </g>
  <g transform="translate(700, 100)">
    <circle cx="0" cy="50" r="10" fill="#d32f2f"/>
    <text x="0" y="80" text-anchor="middle" font-weight="bold">Gen 4</text>
    <text x="0" y="100" text-anchor="middle" font-size="12">Next-Gen (NGFW)</text>
  </g>
  <defs><marker id="arrowR" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#d32f2f" /></marker></defs>
</svg>`,

    'state_table.svg': `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 300">
  <rect width="600" height="300" fill="#e3f2fd" />
  <text x="300" y="40" font-family="Arial" font-weight="bold" text-anchor="middle">Firewall State Table</text>
  
  <g transform="translate(50, 70)">
    <rect x="0" y="0" width="500" height="30" fill="#1976d2"/>
    <text x="10" y="20" fill="white" font-weight="bold">Source IP</text>
    <text x="110" y="20" fill="white" font-weight="bold">Des IP</text>
    <text x="210" y="20" fill="white" font-weight="bold">Proto</text>
    <text x="310" y="20" fill="white" font-weight="bold">State</text>
    <text x="410" y="20" fill="white" font-weight="bold">Timeout</text>
    
    <rect x="0" y="30" width="500" height="30" fill="white" stroke="#bbdefb"/>
    <text x="10" y="50">192.168.1.5</text> <text x="110" y="50">8.8.8.8</text> <text x="210" y="50">TCP</text> <text x="310" y="50">ESTAB</text> <text x="410" y="50">200s</text>
    
    <rect x="0" y="60" width="500" height="30" fill="#f5f5f5" stroke="#bbdefb"/>
    <text x="10" y="80">10.0.0.3</text> <text x="110" y="80">1.1.1.1</text> <text x="210" y="80">UDP</text> <text x="310" y="80">WAIT</text> <text x="410" y="80">30s</text>
  </g>
</svg>`,

    'utm_vs_separate.svg': `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 400">
  <rect width="600" height="400" fill="#fff" />
  <text x="300" y="30" font-family="Arial" font-weight="bold" text-anchor="middle">Separate vs Unified (UTM)</text>
  
  <text x="100" y="80" font-weight="bold">Separate Appliances</text>
  <rect x="50" y="100" width="60" height="40" fill="#ddd" stroke="#333"/> <text x="80" y="125" text-anchor="middle" font-size="10">FW</text>
  <line x1="110" y1="120" x2="150" y2="120" stroke="black"/>
  <rect x="150" y="100" width="60" height="40" fill="#ddd" stroke="#333"/> <text x="180" y="125" text-anchor="middle" font-size="10">IPS</text>
  <line x1="210" y1="120" x2="250" y2="120" stroke="black"/>
  <rect x="250" y="100" width="60" height="40" fill="#ddd" stroke="#333"/> <text x="280" y="125" text-anchor="middle" font-size="10">AV</text>
  <line x1="310" y1="120" x2="350" y2="120" stroke="black"/>
  <rect x="350" y="100" width="60" height="40" fill="#ddd" stroke="#333"/> <text x="380" y="125" text-anchor="middle" font-size="10">Web</text>

  <text x="100" y="230" font-weight="bold">UTM (Unified)</text>
  <rect x="50" y="250" width="200" height="100" fill="#ffecb3" stroke="#ff8f00" stroke-width="2"/>
  <text x="150" y="275" text-anchor="middle" font-weight="bold">UTM Appliance</text>
  <text x="150" y="300" text-anchor="middle" font-size="10">Firewall + IPS + AV + Web</text>
  <text x="150" y="320" text-anchor="middle" font-size="10">One Device, One Console</text>
</svg>`,

    'firewall_ha.svg': `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 300">
  <rect width="600" height="300" fill="#e8eaf6" />
  <text x="300" y="30" font-family="Arial" font-weight="bold" text-anchor="middle">Firewall High Availability</text>
  
  <rect x="100" y="100" width="120" height="60" fill="#4caf50" stroke="#1b5e20"/>
  <text x="160" y="130" fill="white" text-anchor="middle" font-weight="bold">Primary (Active)</text>
  
  <rect x="380" y="100" width="120" height="60" fill="#bdbdbd" stroke="#616161"/>
  <text x="440" y="130" fill="white" text-anchor="middle" font-weight="bold">Secondary (Passive)</text>
  
  <path d="M 220 130 L 380 130" stroke="#d32f2f" stroke-width="2" stroke-dasharray="5,5"/>
  <text x="300" y="120" fill="#d32f2f" text-anchor="middle" font-size="12">Heartbeat / Sync</text>
  
  <line x1="160" y1="160" x2="160" y2="250" stroke="green" stroke-width="4" marker-end="url(#arrowHA)"/>
  <text x="160" y="270" text-anchor="middle">Traffic</text>
  
  <defs><marker id="arrowHA" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="green" /></marker></defs>
</svg>`,

    'mitm_attack.svg': `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 300">
  <rect width="600" height="300" fill="#212121" />
  <text x="300" y="40" fill="white" font-family="Arial" font-weight="bold" text-anchor="middle">Man-in-the-Middle Attack</text>
  
  <rect x="50" y="120" width="80" height="60" fill="#42a5f5"/> <text x="90" y="155" fill="white" text-anchor="middle">Client</text>
  <rect x="470" y="120" width="80" height="60" fill="#66bb6a"/> <text x="510" y="155" fill="white" text-anchor="middle">Server</text>
  
  <rect x="260" y="120" width="80" height="60" fill="#ef5350"/> <text x="300" y="155" fill="white" text-anchor="middle">Attacker</text>
  
  <path d="M 130 150 L 260 150" stroke="white" stroke-width="2" marker-end="url(#arrowW)"/>
  <path d="M 340 150 L 470 150" stroke="white" stroke-width="2" marker-end="url(#arrowW)"/>
  
  <text x="200" y="140" fill="#ccc" font-size="10">Traffic Intercepted</text>
  <text x="400" y="140" fill="#ccc" font-size="10">Traffic Relayed</text>
  
  <defs><marker id="arrowW" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="white" /></marker></defs>
</svg>`,

    'snort_rules.svg': `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 200">
  <rect width="800" height="200" fill="#fff" stroke="#333"/>
  <text x="400" y="40" font-family="Courier New" font-weight="bold" text-anchor="middle" font-size="18">alert tcp any any -> 192.168.1.0/24 80 (msg:"Attack"; sid:1001;)</text>
  
  <line x1="80" y1="50" x2="100" y2="100" stroke="red"/> <text x="100" y="120" fill="red">Action</text>
  <line x1="140" y1="50" x2="160" y2="100" stroke="blue"/> <text x="160" y="120" fill="blue">Proto</text>
  <line x1="200" y1="50" x2="220" y2="100" stroke="green"/> <text x="220" y="120" fill="green">Src IP</text>
  <line x1="300" y1="50" x2="320" y2="100" stroke="orange"/> <text x="320" y="120" fill="orange">Dir</text>
  <line x1="450" y1="50" x2="470" y2="100" stroke="purple"/> <text x="470" y="120" fill="purple">Dst IP/Port</text>
  <line x1="650" y1="50" x2="670" y2="100" stroke="#333"/> <text x="670" y="120">Options</text>
</svg>`,

    'uml_generalization.svg': `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300">
  <rect width="400" height="300" fill="#fff" />
  <text x="200" y="30" font-family="Arial" font-weight="bold" text-anchor="middle">Generalization (Inheritance)</text>
  
  <!-- Parent -->
  <rect x="150" y="50" width="100" height="50" fill="#fff3e0" stroke="#e65100"/>
  <text x="200" y="80" text-anchor="middle">Vehicle</text>
  
  <!-- Children -->
  <rect x="50" y="200" width="100" height="50" fill="#fff3e0" stroke="#e65100"/>
  <text x="100" y="230" text-anchor="middle">Car</text>
  
  <rect x="250" y="200" width="100" height="50" fill="#fff3e0" stroke="#e65100"/>
  <text x="300" y="230" text-anchor="middle">Bike</text>
  
  <!-- Lines with hollow triangle -->
  <path d="M 100 200 L 100 150 L 200 150 L 200 115" stroke="#e65100" stroke-width="1.5" fill="none"/>
  <path d="M 300 200 L 300 150 L 200 150" stroke="#e65100" stroke-width="1.5" fill="none"/>
  
  <path d="M 200 100 L 190 115 L 210 115 Z" fill="white" stroke="#e65100" stroke-width="1.5"/>
</svg>`,

    'state_notation.svg': `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 200">
  <rect width="300" height="200" fill="#fff" />
  <rect x="50" y="50" width="200" height="120" rx="15" ry="15" fill="#f3e5f5" stroke="#8e24aa" stroke-width="2"/>
  
  <text x="150" y="80" font-family="Arial" font-weight="bold" text-anchor="middle">State Name</text>
  <line x1="50" y1="90" x2="250" y2="90" stroke="#8e24aa"/>
  
  <text x="60" y="115" font-family="Arial" font-size="12">entry / action()</text>
  <text x="60" y="135" font-family="Arial" font-size="12">do / activity()</text>
  <text x="60" y="155" font-family="Arial" font-size="12">exit / cleanup()</text>
</svg>`,

    'layered_architecture.svg': `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400">
  <rect width="400" height="400" fill="#fff" />
  <text x="200" y="30" font-family="Arial" font-weight="bold" text-anchor="middle">Layered Architecture</text>
  
  <rect x="100" y="60" width="200" height="60" fill="#bbdefb" stroke="#1976d2"/>
  <text x="200" y="95" text-anchor="middle" font-weight="bold">Presentation Layer</text>

  <path d="M 200 120 L 200 160" stroke="#333" stroke-width="2" marker-end="url(#arrowD)"/>
  
  <rect x="100" y="160" width="200" height="60" fill="#c8e6c9" stroke="#388e3c"/>
  <text x="200" y="195" text-anchor="middle" font-weight="bold">Business Logic</text>
  
  <path d="M 200 220 L 200 260" stroke="#333" stroke-width="2" marker-end="url(#arrowD)"/>
  
  <rect x="100" y="260" width="200" height="60" fill="#ffccbc" stroke="#d84315"/>
  <text x="200" y="295" text-anchor="middle" font-weight="bold">Data Access Layer</text>
  
  <defs><marker id="arrowD" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#333" /></marker></defs>
</svg>`,

    'mvc_pattern.svg': `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 400">
  <rect width="500" height="400" fill="#fff" />
  <text x="250" y="30" font-family="Arial" font-weight="bold" text-anchor="middle">MVC Pattern</text>
  
  <rect x="200" y="50" width="100" height="60" fill="#e1bee7" stroke="#7b1fa2"/>
  <text x="250" y="85" text-anchor="middle" font-weight="bold">Model</text>
  
  <rect x="50" y="250" width="100" height="60" fill="#b2dfdb" stroke="#00796b"/>
  <text x="100" y="285" text-anchor="middle" font-weight="bold">View</text>
  
  <rect x="350" y="250" width="100" height="60" fill="#ffecb3" stroke="#ffa000"/>
  <text x="400" y="285" text-anchor="middle" font-weight="bold">Controller</text>
  
  <!-- Arrows -->
  <path d="M 150 280 L 350 280" stroke="#333" stroke-width="2"/> <!-- V <-> C (indirect) used by User -->
  <line x1="250" y1="110" x2="100" y2="250" stroke="#333" stroke-width="2" marker-end="url(#arrowM)"/> <!-- M updates V -->
  <line x1="400" y1="250" x2="300" y2="110" stroke="#333" stroke-width="2" marker-end="url(#arrowM)"/> <!-- C updates M -->
  
  <text x="130" y="180" font-size="10" transform="rotate(-45 130,180)">Updates</text>
  <text x="370" y="180" font-size="10" transform="rotate(45 370,180)">Manipulates</text>
  
  <defs><marker id="arrowM" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#333" /></marker></defs>
</svg>`,

    'eisenhower_matrix.svg': `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500">
  <rect width="500" height="500" fill="#fff" />
  <text x="250" y="40" font-family="Arial" font-weight="bold" text-anchor="middle">Eisenhower Matrix</text>
  
  <line x1="50" y1="80" x2="450" y2="80" stroke="black"/>
  <line x1="50" y1="275" x2="450" y2="275" stroke="black"/>
  <line x1="250" y1="80" x2="250" y2="470" stroke="black"/>
  
  <text x="150" y="70" text-anchor="middle" font-weight="bold">Urgent</text>
  <text x="350" y="70" text-anchor="middle" font-weight="bold">Not Urgent</text>
  
  <text x="30" y="180" transform="rotate(-90 30,180)" text-anchor="middle" font-weight="bold">Important</text>
  <text x="30" y="380" transform="rotate(-90 30,380)" text-anchor="middle" font-weight="bold">Not Important</text>
  
  <rect x="60" y="90" width="180" height="180" fill="#ffcdd2"/>
  <text x="150" y="180" text-anchor="middle" font-weight="bold" font-size="20">DO</text>
  
  <rect x="260" y="90" width="180" height="180" fill="#c8e6c9"/>
  <text x="350" y="180" text-anchor="middle" font-weight="bold" font-size="20">SCHEDULE</text>
  
  <rect x="60" y="285" width="180" height="180" fill="#fff9c4"/>
  <text x="150" y="375" text-anchor="middle" font-weight="bold" font-size="20">DELEGATE</text>
  
  <rect x="260" y="285" width="180" height="180" fill="#f5f5f5"/>
  <text x="350" y="375" text-anchor="middle" font-weight="bold" font-size="20">ELIMINATE</text>
</svg>`,

    'circle_of_influence.svg': `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500">
  <rect width="500" height="500" fill="#fff" />
  <text x="250" y="40" font-family="Arial" font-weight="bold" text-anchor="middle">Circle of Influence</text>
  
  <circle cx="250" cy="250" r="180" fill="#e0e0e0" stroke="#9e9e9e" stroke-width="2"/>
  <text x="250" y="120" text-anchor="middle" fill="#616161" font-weight="bold">Circle of Concern</text>
  <text x="250" y="140" text-anchor="middle" font-size="12" fill="#616161">(Weather, Politics, Others)</text>

  <circle cx="250" cy="250" r="100" fill="#81c784" stroke="#2e7d32" stroke-width="2"/>
  <text x="250" y="240" text-anchor="middle" fill="#1b5e20" font-weight="bold">Circle of Influence</text>
  <text x="250" y="260" text-anchor="middle" font-size="12" fill="#1b5e20">(My Attitude, My Actions)</text>
</svg>`,

    'tuckman_model.svg': `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 300">
  <rect width="800" height="300" fill="#f3e5f5" />
  <text x="400" y="40" font-family="Arial" font-weight="bold" text-anchor="middle">Tuckman's Team Stages</text>
  
  <polyline points="50,250 150,200 300,230 500,100 700,50" fill="none" stroke="#673ab7" stroke-width="4"/>
  
  <circle cx="50" cy="250" r="10" fill="#673ab7"/> <text x="50" y="280" text-anchor="middle">Forming</text>
  <circle cx="150" cy="200" r="10" fill="#673ab7"/> <text x="150" y="180" text-anchor="middle">Storming</text>
  <circle cx="300" cy="230" r="10" fill="#673ab7"/> <text x="300" y="260" text-anchor="middle">Norming</text>
  <circle cx="500" cy="100" r="10" fill="#673ab7"/> <text x="500" y="130" text-anchor="middle">Performing</text>
  <circle cx="700" cy="50" r="10" fill="#673ab7"/> <text x="700" y="80" text-anchor="middle">Adjourning</text>
  
  <text x="10" y="150" transform="rotate(-90 20,150)" font-weight="bold" fill="#673ab7">Performance</text>
</svg>`
};

for (const [filename, content] of Object.entries(diagrams)) {
    fs.writeFileSync(path.join(outDir, filename), content.trim());
    console.log(`Generated ${filename}`);
}
