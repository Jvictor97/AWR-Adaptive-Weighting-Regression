from dataloader.custom_loader import CustomLoader
from dataloader.nyu_loader import NYU

# loader = NYU('./data', 'test')
# loader[0]

loader = CustomLoader('./data/custom', 'test')
#print(loader)

for i in range(6):
  loader[i]