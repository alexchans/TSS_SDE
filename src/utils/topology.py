"""Standard amino acid topology: heavy-atom covalent bonds and ideal lengths (Å)."""

BACKBONE_BONDS = {
    ('N', 'CA'):  1.458,
    ('CA', 'C'):  1.524,
    ('C', 'O'):   1.231,
}

PEPTIDE_BOND = ('C', 'N', 1.329)

SIDECHAIN_BONDS = {
    'GLY': {},

    'ALA': {
        ('CA', 'CB'): 1.521,
    },

    'VAL': {
        ('CA', 'CB'):  1.540,
        ('CB', 'CG1'): 1.521,
        ('CB', 'CG2'): 1.521,
    },

    'LEU': {
        ('CA', 'CB'): 1.530,
        ('CB', 'CG'): 1.530,
        ('CG', 'CD1'): 1.521,
        ('CG', 'CD2'): 1.521,
    },

    'ILE': {
        ('CA', 'CB'):  1.540,
        ('CB', 'CG1'): 1.530,
        ('CB', 'CG2'): 1.521,
        ('CG1', 'CD1'): 1.513,
    },

    'PRO': {
        ('CA', 'CB'): 1.530,
        ('CB', 'CG'): 1.492,
        ('CG', 'CD'): 1.503,
        ('CD', 'N'):  1.473,
    },

    'PHE': {
        ('CA', 'CB'):  1.530,
        ('CB', 'CG'):  1.502,
        ('CG', 'CD1'): 1.384,
        ('CG', 'CD2'): 1.384,
        ('CD1', 'CE1'): 1.390,
        ('CD2', 'CE2'): 1.390,
        ('CE1', 'CZ'):  1.382,
        ('CE2', 'CZ'):  1.382,
    },

    'TYR': {
        ('CA', 'CB'):  1.530,
        ('CB', 'CG'):  1.512,
        ('CG', 'CD1'): 1.389,
        ('CG', 'CD2'): 1.389,
        ('CD1', 'CE1'): 1.390,
        ('CD2', 'CE2'): 1.390,
        ('CE1', 'CZ'):  1.382,
        ('CE2', 'CZ'):  1.382,
        ('CZ', 'OH'):   1.376,
    },

    'TRP': {
        ('CA', 'CB'):  1.530,
        ('CB', 'CG'):  1.498,
        ('CG', 'CD1'): 1.365,
        ('CG', 'CD2'): 1.433,
        ('CD1', 'NE1'): 1.374,
        ('NE1', 'CE2'): 1.370,
        ('CD2', 'CE2'): 1.409,
        ('CD2', 'CE3'): 1.398,
        ('CE2', 'CZ2'): 1.394,
        ('CE3', 'CZ3'): 1.382,
        ('CZ2', 'CH2'): 1.368,
        ('CZ3', 'CH2'): 1.400,
    },

    'SER': {
        ('CA', 'CB'): 1.530,
        ('CB', 'OG'): 1.417,
    },

    'THR': {
        ('CA', 'CB'):  1.540,
        ('CB', 'OG1'): 1.433,
        ('CB', 'CG2'): 1.521,
    },

    'CYS': {
        ('CA', 'CB'): 1.530,
        ('CB', 'SG'): 1.808,
    },

    'MET': {
        ('CA', 'CB'): 1.520,
        ('CB', 'CG'): 1.520,
        ('CG', 'SD'): 1.803,
        ('SD', 'CE'): 1.791,
    },

    'ASP': {
        ('CA', 'CB'): 1.530,
        ('CB', 'CG'): 1.516,
        ('CG', 'OD1'): 1.249,
        ('CG', 'OD2'): 1.249,
    },

    'GLU': {
        ('CA', 'CB'): 1.530,
        ('CB', 'CG'): 1.520,
        ('CG', 'CD'): 1.516,
        ('CD', 'OE1'): 1.249,
        ('CD', 'OE2'): 1.249,
    },

    'ASN': {
        ('CA', 'CB'): 1.530,
        ('CB', 'CG'): 1.516,
        ('CG', 'OD1'): 1.231,
        ('CG', 'ND2'): 1.328,
    },

    'GLN': {
        ('CA', 'CB'): 1.530,
        ('CB', 'CG'): 1.520,
        ('CG', 'CD'): 1.516,
        ('CD', 'OE1'): 1.231,
        ('CD', 'NE2'): 1.328,
    },

    'LYS': {
        ('CA', 'CB'): 1.530,
        ('CB', 'CG'): 1.520,
        ('CG', 'CD'): 1.520,
        ('CD', 'CE'): 1.520,
        ('CE', 'NZ'): 1.489,
    },

    'ARG': {
        ('CA', 'CB'): 1.530,
        ('CB', 'CG'): 1.520,
        ('CG', 'CD'): 1.520,
        ('CD', 'NE'): 1.460,
        ('NE', 'CZ'): 1.329,
        ('CZ', 'NH1'): 1.326,
        ('CZ', 'NH2'): 1.326,
    },

    'HIS': {
        ('CA', 'CB'):  1.530,
        ('CB', 'CG'):  1.497,
        ('CG', 'ND1'): 1.371,
        ('CG', 'CD2'): 1.354,
        ('ND1', 'CE1'): 1.319,
        ('CD2', 'NE2'): 1.374,
        ('CE1', 'NE2'): 1.321,
    },
}

# Ordered most-specific to least-specific for matching
_RESIDUE_SIGNATURES = [
    ('TRP', {'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'}),
    ('ARG', {'NE', 'CZ', 'NH1', 'NH2'}),
    ('HIS', {'ND1', 'CD2', 'CE1', 'NE2'}),
    ('TYR', {'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'}),
    ('PHE', {'CD1', 'CD2', 'CE1', 'CE2', 'CZ'}),
    ('LYS', {'CD', 'CE', 'NZ'}),
    ('GLN', {'CD', 'OE1', 'NE2'}),
    ('GLU', {'CD', 'OE1', 'OE2'}),
    ('MET', {'SD', 'CE'}),
    ('ASN', {'CG', 'OD1', 'ND2'}),
    ('ASP', {'CG', 'OD1', 'OD2'}),
    ('LEU', {'CG', 'CD1', 'CD2'}),
    ('ILE', {'CG1', 'CG2', 'CD1'}),
    ('PRO', {'CG', 'CD'}),
    ('VAL', {'CG1', 'CG2'}),
    ('THR', {'OG1', 'CG2'}),
    ('CYS', {'SG'}),
    ('SER', {'OG'}),
    ('ALA', {'CB'}),
    ('GLY', set()),
]


def identify_residue_type(atom_names_in_residue):
    """Identify amino acid type from the set of atom names in a residue."""
    atom_set = set(atom_names_in_residue)
    for resname, signature in _RESIDUE_SIGNATURES:
        if signature.issubset(atom_set):
            return resname
    return 'UNK'


def build_bonds_from_atom_names(atom_names_path):
    """Build covalent bond definitions by parsing atom names, segmenting into residues, and applying standard topology."""
    with open(atom_names_path) as f:
        all_atoms = [line.strip() for line in f if line.strip()]

    # Segment into residues (each starts with backbone 'N')
    residues = []
    current_start = None
    current_atoms = []

    for i, name in enumerate(all_atoms):
        if name == 'N':
            if current_atoms:
                residues.append((current_start, current_atoms))
            current_start = i
            current_atoms = [name]
        else:
            current_atoms.append(name)
    if current_atoms:
        residues.append((current_start, current_atoms))

    # Build bonds
    bond_defs = []

    for res_idx, (start, atom_names) in enumerate(residues):
        restype = identify_residue_type(atom_names)
        name_to_global = {name: start + idx for idx, name in enumerate(atom_names)}

        # Backbone bonds
        for (a1, a2), length in BACKBONE_BONDS.items():
            if a1 in name_to_global and a2 in name_to_global:
                bond_defs.append((name_to_global[a1], name_to_global[a2], length))

        # Side-chain bonds
        if restype in SIDECHAIN_BONDS:
            for (a1, a2), length in SIDECHAIN_BONDS[restype].items():
                if a1 in name_to_global and a2 in name_to_global:
                    bond_defs.append((name_to_global[a1], name_to_global[a2], length))
        elif restype == 'UNK':
            print(f"  Warning: Unknown residue at position {res_idx} ({atom_names})")

        # Peptide bond to next residue
        if res_idx < len(residues) - 1:
            next_start, next_atoms = residues[res_idx + 1]
            next_name_to_global = {name: next_start + idx for idx, name in enumerate(next_atoms)}
            c_name, n_name, length = PEPTIDE_BOND
            if c_name in name_to_global and n_name in next_name_to_global:
                bond_defs.append((name_to_global[c_name], next_name_to_global[n_name], length))

    return bond_defs, residues
