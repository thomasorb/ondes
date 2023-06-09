sudo dpkg --purge --force-depends pulseaudio alsa-base alsa-utils
sudo apt --fix-broken install
pulseaudio -k && sudo alsa force-reload
