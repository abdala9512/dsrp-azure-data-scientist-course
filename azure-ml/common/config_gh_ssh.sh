cd
cd .ssh
eval "$(ssh-agent -s)"
exec ssh-agent bash
chmod 400 dsrp_llave # fist time
ssh-add dsrp_llave 
ssh -vT git@github.com