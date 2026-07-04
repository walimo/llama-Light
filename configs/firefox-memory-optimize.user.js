// Firefox memory optimization — minimal RAM usage for tabs + saved logins only
// Install: copy to ~/.config/mozilla/firefox/<profile>/user.js and restart Firefox

// Single-process mode — all tabs share one process (saves ~2 GB RAM)
user_pref("dom.ipc.processCount", 1);
user_pref("dom.ipc.processCount.tablet", 1);
user_pref("dom.ipc.tabs.disabled", true);
user_pref("browser.tabs.remote.separateOriginPrinciples", false);
user_pref("browser.tabs.remote.separateProcessPerSite", false);
user_pref("browser.tabs.remote.separateUserContextPerSite", false);

// Minimal cache
user_pref("browser.cache.disk.size", 1024);
user_pref("browser.cache.memory.capacity", 512);

// Disable bloat — no telemetry, no newtab stories, no preload
user_pref("toolkit.telemetry.enabled", false);
user_pref("browser.newtabpage.activity-stream.telemetry", false);
user_pref("browser.ping-centre.telemetry", false);
user_pref("datareporting.healthreport.uploadEnabled", false);
user_pref("datareporting.policy.datasubmissionEnabled", false);
user_pref("browser.newtab.preload", false);
user_pref("browser.newtabpage.activity-stream.feeds.topsites", false);
user_pref("browser.newtabpage.activity-stream.feeds.system.topstories", false);

// Session — don't restore on crash, limit saved tabs
user_pref("browser.sessionstore.max_tabs_val", 10);
user_pref("browser.sessionstore.resume_from_crash", false);

// Reduce font cache
user_pref("gfx.font_cache.size", 50);
