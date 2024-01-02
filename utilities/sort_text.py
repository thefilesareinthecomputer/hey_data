'''takes a python list and prints an alphabetically sorted version'''

science_websites = [
    'https://arxiv.org/',
    'https://blog.scienceopen.com/',
    'https://clinicaltrials.gov/',
    'https://doaj.org/',
    'https://osf.io/preprints/psyarxiv',
    'https://plos.org/',
    'https://pubmed.ncbi.nlm.nih.gov/',
    'https://scholar.google.com/',
    'https://www.amjmed.com/',
    'https://www.cdc.gov/',
    'https://www.cell.com/',
    'https://www.drugs.com/',
    'https://www.health.harvard.edu/',
    'https://www.health.harvard.edu/',
    'https://www.mayoclinic.org/',
    'https://www.mayoclinic.org/',
    'https://www.medicinenet.com/',
    'https://www.medlineplus.gov/',
    'https://www.nature.com/',
    'https://www.ncbi.nlm.nih.gov/pmc',
    'https://www.nejm.org/',
    'https://www.nhs.uk/',
    'https://www.nih.gov/',
    'https://www.nlm.nih.gov/',
    'https://www.safemedication.com/',
    'https://www.science.gov/',
    'https://www.science.org/',
    'https://www.semanticscholar.org/',
    'https://zenodo.org/',
    ]

list_name = science_websites

list_name.sort()

print("# Sorted")
print("sorted_list = [")
for word in list_name[:-1]:
    print(f"    '{word}',")
print(f"    '{list_name[-1]}'")
print("]")