export interface AppConfig {
  pageTitle: string;
  pageDescription: string;
  companyName: string;

  supportsChatInput: boolean;
  supportsVideoInput: boolean;
  supportsScreenShare: boolean;
  isPreConnectBufferEnabled: boolean;

  logo: string;
  startButtonText: string;
  accent?: string;
  logoDark?: string;
  accentDark?: string;

  // for LiveKit Cloud Sandbox
  sandboxId?: string;
  agentName?: string;
}

export const APP_CONFIG_DEFAULTS: AppConfig = {
  companyName: 'Flipkart',
  pageTitle: 'FlipMin',
  pageDescription: 'A voice agent built for Flipkart customers.',

  supportsChatInput: true,
  supportsVideoInput: true,
  supportsScreenShare: true,
  isPreConnectBufferEnabled: true,

  logo: '/icon.png',
  accent: '#002cf2',
  logoDark: '/icon.png',
  accentDark: '#ffffff',
  startButtonText: 'Order Now',

  // for LiveKit Cloud Sandbox
  sandboxId: undefined,
  agentName: undefined,
};
