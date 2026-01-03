const fs = require('fs');
const path = require('path');

const outDir = path.join(__dirname, 'public/diagrams');

const diagrams = {
    'id3_flowchart.svg': `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 500">
  <rect width="600" height="500" fill="#e8f5e9" />
  <text x="300" y="30" font-family="Arial" font-weight="bold" text-anchor="middle">ID3 Algorithm</text>
  
  <rect x="250" y="50" width="100" height="40" rx="20" fill="#66bb6a" stroke="#2e7d32"/>
  <text x="300" y="75" text-anchor="middle" font-weight="bold" fill="white">Start</text>
  <line x1="300" y1="90" x2="300" y2="130" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>

  <polygon points="300,130 350,160 300,190 250,160" fill="#fff" stroke="#333"/>
  <text x="300" y="165" text-anchor="middle" font-size="10">Homogeneous?</text>
  
  <text x="360" y="160" font-size="10">Yes</text>
  <line x1="350" y1="160" x2="400" y2="160" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
  <rect x="400" y="140" width="80" height="40" fill="#81c784" stroke="#2e7d32"/>
  <text x="440" y="165" text-anchor="middle" font-size="10">Create Leaf</text>

  <text x="300" y="205" font-size="10">No</text>
  <line x1="300" y1="190" x2="300" y2="230" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>

  <rect x="220" y="230" width="160" height="40" fill="#fff" stroke="#333"/>
  <text x="300" y="255" text-anchor="middle" font-size="10">Calc Entropy & Gain</text>
  <line x1="300" y1="270" x2="300" y2="310" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>

  <rect x="220" y="310" width="160" height="40" fill="#fff" stroke="#333"/>
  <text x="300" y="335" text-anchor="middle" font-size="10">Select Best Attribute</text>
  <line x1="300" y1="350" x2="300" y2="390" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>

  <text x="300" y="420" text-anchor="middle" font-size="12">Recurse for each Branch</text>
  <defs><marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#333" /></marker></defs>
</svg>`,

    'entropy_calc.svg': `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 300">
  <rect width="600" height="300" fill="#fff3e0" />
  <text x="300" y="30" font-family="Arial" font-weight="bold" text-anchor="middle">Entropy & Information Gain</text>
  
  <text x="150" y="60" text-anchor="middle" font-weight="bold">High Entropy (Impure)</text>
  <circle cx="150" cy="120" r="50" fill="#fff" stroke="#333"/>
  <text x="140" y="110" font-size="20">●</text> <text x="160" y="110" font-size="20" fill="red">✖</text>
  <text x="130" y="130" font-size="20" fill="red">✖</text> <text x="150" y="130" font-size="20">●</text>
  <text x="170" y="130" font-size="20">●</text> <text x="150" y="150" font-size="20" fill="red">✖</text>
  <text x="150" y="190" text-anchor="middle">H(S) ≈ 1.0</text>

  <line x1="220" y1="120" x2="380" y2="120" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
  <text x="300" y="110" text-anchor="middle">Split by Attribute</text>

  <text x="450" y="60" text-anchor="middle" font-weight="bold">Low Entropy (Pure)</text>
  <circle cx="450" cy="90" r="30" fill="#fff" stroke="#333"/>
  <text x="450" y="95" text-anchor="middle">● ● ●</text>
  <text x="500" y="90" fill="green">H=0</text>

  <circle cx="450" cy="170" r="30" fill="#fff" stroke="#333"/>
  <text x="450" y="175" text-anchor="middle" fill="red">✖ ✖ ✖</text>
  <text x="500" y="170" fill="green">H=0</text>
  
  <text x="300" y="250" text-anchor="middle" font-weight="bold">Gain = H(Parent) - WeightedAvg(H(Children))</text>
</svg>`,

    'inductive_vs_analytical.svg': `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 400">
  <rect width="600" height="400" fill="#e3f2fd" />
  <text x="300" y="30" font-family="Arial" font-weight="bold" text-anchor="middle">Inductive vs Analytical Learning</text>
  
  <!-- Inductive -->
  <g transform="translate(50, 60)">
    <rect x="0" y="0" width="220" height="300" fill="#fff" stroke="#1565c0" rx="10"/>
    <text x="110" y="30" text-anchor="middle" font-weight="bold" fill="#1565c0">Inductive Learning</text>
    <text x="110" y="50" text-anchor="middle" font-size="12">(e.g., Neural Nets, Trees)</text>
    
    <text x="110" y="100" text-anchor="middle">Training Examples</text>
    <line x1="110" y1="110" x2="110" y2="150" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    <text x="110" y="170" text-anchor="middle" font-weight="bold">General Hypothesis</text>
    <text x="110" y="250" text-anchor="middle" font-size="12" fill="#555">"Knowledge from Data"</text>
  </g>

  <!-- Analytical -->
  <g transform="translate(330, 60)">
    <rect x="0" y="0" width="220" height="300" fill="#fff" stroke="#c62828" rx="10"/>
    <text x="110" y="30" text-anchor="middle" font-weight="bold" fill="#c62828">Analytical Learning</text>
    <text x="110" y="50" text-anchor="middle" font-size="12">(e.g., Explanation Based)</text>
    
    <text x="60" y="100" text-anchor="middle" font-size="12">Examples</text>
    <text x="160" y="100" text-anchor="middle" font-size="12">Prior Knowledge</text>
    <line x1="60" y1="110" x2="100" y2="150" stroke="#333" stroke-width="2"/>
    <line x1="160" y1="110" x2="120" y2="150" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    
    <text x="110" y="170" text-anchor="middle" font-weight="bold">Generalized Rules</text>
    <text x="110" y="250" text-anchor="middle" font-size="12" fill="#555">"Knowledge Guided"</text>
  </g>
</svg>`,

    'ebg_process.svg': `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 300">
  <rect width="600" height="300" fill="#f3e5f5" />
  <text x="300" y="30" font-family="Arial" font-weight="bold" text-anchor="middle">Explanation-Based Generalization (EBG)</text>
  
  <rect x="50" y="60" width="100" height="40" fill="#e1bee7" stroke="#4a148c"/> <text x="100" y="85" text-anchor="middle" font-size="12">Domain Theory</text>
  <rect x="50" y="120" width="100" height="40" fill="#e1bee7" stroke="#4a148c"/> <text x="100" y="145" text-anchor="middle" font-size="12">Target Concept</text>
  <rect x="50" y="180" width="100" height="40" fill="#e1bee7" stroke="#4a148c"/> <text x="100" y="205" text-anchor="middle" font-size="12">Training Ex</text>
  
  <path d="M 160 80 L 220 140" stroke="#333" stroke-width="2"/>
  <path d="M 160 140 L 220 140" stroke="#333" stroke-width="2"/>
  <path d="M 160 200 L 220 140" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>

  <rect x="230" y="100" width="140" height="80" fill="#fff" stroke="#333"/>
  <text x="300" y="130" text-anchor="middle" font-weight="bold">Explanation</text>
  <text x="300" y="150" text-anchor="middle" font-size="10">(Proof Tree)</text>
  
  <line x1="380" y1="140" x2="440" y2="140" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
  
  <rect x="450" y="100" width="120" height="80" fill="#ce93d8" stroke="#4a148c"/>
  <text x="510" y="135" text-anchor="middle" font-weight="bold">General Rule</text>
  <text x="510" y="155" text-anchor="middle" font-size="10">Operationalized</text>
</svg>`,

    'qlearning_process.svg': `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 400">
  <rect width="600" height="400" fill="#fff8e1" />
  <text x="300" y="30" font-family="Arial" font-weight="bold" text-anchor="middle">Q-Learning Process</text>
  
  <!-- Agent -->
  <circle cx="150" cy="200" r="60" fill="#ffecb3" stroke="#ff6f00" stroke-width="2"/>
  <text x="150" y="205" text-anchor="middle" font-weight="bold">Agent</text>
  
  <!-- Environment -->
  <rect x="350" y="140" width="200" height="120" fill="#b2dfdb" stroke="#00695c" stroke-width="2"/>
  <text x="450" y="205" text-anchor="middle" font-weight="bold">Environment</text>
  
  <!-- Arrows -->
  <path d="M 180 160 Q 300 100 380 140" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
  <text x="280" y="120" text-anchor="middle" font-weight="bold">Action (a)</text>
  
  <path d="M 380 260 Q 300 300 180 240" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
  <text x="280" y="300" text-anchor="middle" font-weight="bold">State (s'), Reward (r)</text>
  
  <!-- Formula -->
  <rect x="50" y="340" width="500" height="40" fill="#fff" stroke="#333"/>
  <text x="300" y="365" text-anchor="middle" font-family="monospace">Q(s,a) ← Q(s,a) + α[r + γ·maxQ(s',a') - Q(s,a)]</text>
</svg>`,

    'utm_vs_ngfw.svg': `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 300">
  <rect width="600" height="300" fill="#e0f2f1" />
  <text x="300" y="30" font-family="Arial" font-weight="bold" text-anchor="middle">UTM vs NGFW</text>
  
  <!-- UTM -->
  <g transform="translate(50, 60)">
    <rect x="0" y="0" width="220" height="200" fill="#fff" stroke="#00695c"/>
    <text x="110" y="30" text-anchor="middle" font-weight="bold" fill="#00695c">UTM</text>
    <text x="110" y="50" text-anchor="middle" font-size="10">Unified Threat Management</text>
    
    <rect x="30" y="70" width="160" height="100" fill="#b2dfdb"/>
    <text x="110" y="100" text-anchor="middle">Firewall + VPN</text>
    <text x="110" y="120" text-anchor="middle">IPS + AV + Web Filter</text>
    <text x="110" y="140" text-anchor="middle">Email Security</text>
    
    <text x="110" y="190" text-anchor="middle" font-size="12" font-style="italic">"All-in-One Box"</text>
  </g>

  <!-- NGFW -->
  <g transform="translate(330, 60)">
    <rect x="0" y="0" width="220" height="200" fill="#fff" stroke="#0277bd"/>
    <text x="110" y="30" text-anchor="middle" font-weight="bold" fill="#0277bd">NGFW</text>
    <text x="110" y="50" text-anchor="middle" font-size="10">Next-Gen Firewall</text>
    
    <rect x="30" y="70" width="160" height="100" fill="#b3e5fc"/>
    <text x="110" y="100" text-anchor="middle">Deep Packet Inspection</text>
    <text x="110" y="120" text-anchor="middle">Application ID</text>
    <text x="110" y="140" text-anchor="middle">User ID</text>
    
    <text x="110" y="190" text-anchor="middle" font-size="12" font-style="italic">"Granular Control"</text>
  </g>
</svg>`,

    'ood_process.svg': `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 200">
  <rect width="600" height="200" fill="#f3e5f5" />
  <text x="300" y="30" font-family="Arial" font-weight="bold" text-anchor="middle">Object-Oriented Development Cycle</text>
  
  <circle cx="100" cy="100" r="40" fill="#e1bee7" stroke="#4a148c"/> <text x="100" y="105" text-anchor="middle" font-size="12">Analysis</text>
  <line x1="140" y1="100" x2="210" y2="100" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
  
  <circle cx="250" cy="100" r="40" fill="#ce93d8" stroke="#4a148c"/> <text x="250" y="105" text-anchor="middle" font-size="12">Design</text>
  <line x1="290" y1="100" x2="360" y2="100" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
  
  <circle cx="400" cy="100" r="40" fill="#ba68c8" stroke="#4a148c"/> <text x="400" y="105" text-anchor="middle" font-size="12">Implement</text>
  <line x1="440" y1="100" x2="510" y2="100" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
  
  <circle cx="550" cy="100" r="40" fill="#ab47bc" stroke="#4a148c"/> <text x="550" y="105" text-anchor="middle" font-size="12">Test</text>
  
  <path d="M 550 140 Q 325 220 100 140" fill="none" stroke="#333" stroke-dasharray="5,5" marker-end="url(#arrow)"/>
  <text x="325" y="190" text-anchor="middle" font-size="12">Iterate</text>
</svg>`,

    'association_multiplicity.svg': `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 200">
  <rect width="400" height="200" fill="#fff" />
  <text x="200" y="30" font-family="Arial" font-weight="bold" text-anchor="middle">Association & Multiplicity</text>
  
  <rect x="50" y="80" width="100" height="50" fill="#fff" stroke="#333"/>
  <text x="100" y="110" text-anchor="middle" font-weight="bold">Person</text>
  
  <line x1="150" y1="105" x2="250" y2="105" stroke="#333" stroke-width="2"/>
  
  <rect x="250" y="80" width="100" height="50" fill="#fff" stroke="#333"/>
  <text x="300" y="110" text-anchor="middle" font-weight="bold">Car</text>
  
  <text x="160" y="100" font-size="12">1</text>
  <text x="240" y="100" font-size="12">0..*</text>
  <text x="200" y="125" text-anchor="middle" font-size="12" font-style="italic">owns ></text>
</svg>`,

    'state_diagram_order.svg': `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 200">
  <rect width="600" height="200" fill="#fff" />
  <text x="300" y="30" font-family="Arial" font-weight="bold" text-anchor="middle">Order State Diagram</text>
  
  <circle cx="50" cy="100" r="15" fill="#333"/>
  <line x1="65" y1="100" x2="110" y2="100" stroke="#333" marker-end="url(#arrow)"/>
  
  <rect x="110" y="80" width="80" height="40" rx="10" fill="#fff9c4" stroke="#fbc02d"/>
  <text x="150" y="105" text-anchor="middle">Placed</text>
  <line x1="190" y1="100" x2="240" y2="100" stroke="#333" marker-end="url(#arrow)"/>
  
  <rect x="240" y="80" width="80" height="40" rx="10" fill="#fff9c4" stroke="#fbc02d"/>
  <text x="280" y="105" text-anchor="middle">Paid</text>
  <line x1="320" y1="100" x2="370" y2="100" stroke="#333" marker-end="url(#arrow)"/>
  
  <rect x="370" y="80" width="80" height="40" rx="10" fill="#fff9c4" stroke="#fbc02d"/>
  <text x="410" y="105" text-anchor="middle">Shipped</text>
  <line x1="450" y1="100" x2="500" y2="100" stroke="#333" marker-end="url(#arrow)"/>
  
  <rect x="500" y="80" width="80" height="40" rx="10" fill="#c8e6c9" stroke="#388e3c"/>
  <text x="540" y="105" text-anchor="middle">Delivered</text>
</svg>`,

    'sequence_structure.svg': `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300">
  <rect width="400" height="300" fill="#fff" />
  <text x="200" y="30" font-family="Arial" font-weight="bold" text-anchor="middle">Sequence Diagram Structure</text>
  
  <rect x="50" y="50" width="80" height="40" fill="#eee" stroke="#333"/>
  <text x="90" y="75" text-anchor="middle">:ObjectA</text>
  <line x1="90" y1="90" x2="90" y2="280" stroke="#333" stroke-dasharray="5,5"/>
  
  <rect x="250" y="50" width="80" height="40" fill="#eee" stroke="#333"/>
  <text x="290" y="75" text-anchor="middle">:ObjectB</text>
  <line x1="290" y1="90" x2="290" y2="280" stroke="#333" stroke-dasharray="5,5"/>
  
  <!-- Message -->
  <line x1="90" y1="120" x2="285" y2="120" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
  <text x="190" y="115" text-anchor="middle">message()</text>
  
  <!-- Return -->
  <line x1="290" y1="160" x2="95" y2="160" stroke="#333" stroke-width="2" stroke-dasharray="5,5" marker-end="url(#arrow)"/>
  <text x="190" y="155" text-anchor="middle">return val</text>
  
  <!-- Activation -->
  <rect x="285" y="120" width="10" height="40" fill="#ddd" stroke="#333"/>
</svg>`,

    'collaboration_diagram.svg': `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300">
  <rect width="400" height="300" fill="#fff" />
  <text x="200" y="30" font-family="Arial" font-weight="bold" text-anchor="middle">Collaboration Diagram</text>
  
  <rect x="50" y="120" width="80" height="40" fill="#eee" stroke="#333"/>
  <text x="90" y="145" text-anchor="middle">:Sender</text>
  
  <line x1="130" y1="140" x2="270" y2="140" stroke="#333" stroke-width="2"/>
  
  <rect x="270" y="120" width="80" height="40" fill="#eee" stroke="#333"/>
  <text x="310" y="145" text-anchor="middle">:Receiver</text>
  
  <text x="200" y="130" text-anchor="middle">1: call() →</text>
  <text x="200" y="160" text-anchor="middle">← 2: return</text>
</svg>`,

    'solid_principles.svg': `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 200">
  <rect width="600" height="200" fill="#f5f5f5" />
  <text x="300" y="30" font-family="Arial" font-weight="bold" text-anchor="middle">SOLID Principles</text>
  
  <g transform="translate(30, 60)">
    <rect width="100" height="100" fill="#ef9a9a" rx="10"/>
    <text x="50" y="50" text-anchor="middle" font-size="40" font-weight="bold" fill="#fff">S</text>
    <text x="50" y="80" text-anchor="middle" font-size="10" fill="#fff">SRP</text>
  </g>
  <g transform="translate(140, 60)">
    <rect width="100" height="100" fill="#ce93d8" rx="10"/>
    <text x="50" y="50" text-anchor="middle" font-size="40" font-weight="bold" fill="#fff">O</text>
    <text x="50" y="80" text-anchor="middle" font-size="10" fill="#fff">OCP</text>
  </g>
  <g transform="translate(250, 60)">
    <rect width="100" height="100" fill="#90caf9" rx="10"/>
    <text x="50" y="50" text-anchor="middle" font-size="40" font-weight="bold" fill="#fff">L</text>
    <text x="50" y="80" text-anchor="middle" font-size="10" fill="#fff">LSP</text>
  </g>
  <g transform="translate(360, 60)">
    <rect width="100" height="100" fill="#a5d6a7" rx="10"/>
    <text x="50" y="50" text-anchor="middle" font-size="40" font-weight="bold" fill="#fff">I</text>
    <text x="50" y="80" text-anchor="middle" font-size="10" fill="#fff">ISP</text>
  </g>
  <g transform="translate(470, 60)">
    <rect width="100" height="100" fill="#fff59d" rx="10"/>
    <text x="50" y="50" text-anchor="middle" font-size="40" font-weight="bold" fill="#fbc02d">D</text>
    <text x="50" y="80" text-anchor="middle" font-size="10" fill="#fbc02d">DIP</text>
  </g>
</svg>`
};

for (const [filename, content] of Object.entries(diagrams)) {
    fs.writeFileSync(path.join(outDir, filename), content.trim());
    console.log(`Generated ${filename}`);
}
