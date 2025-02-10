from ..base_downloader import BaseModelDownloader, get_model_dirs
from ..download_utils import DownloadManager

class CivitAIDownloader(BaseModelDownloader):     
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {       
                "model_id": ("STRING", {"multiline": False, "default": "360292"}),
                "token_id": ("STRING", {"multiline": False, "default": "token_here"}),
                "save_dir": (get_model_dirs(),),
                "filename": ("STRING", {"multiline": False, "default": "name_here"}),
                "format": ([".safetensors", ".ckpt", ".onnx", ".gguf", ".bin", ".pt", ""])
            },
            "hidden": {
                "node_id": "UNIQUE_ID"
            }
        }
        
    FUNCTION = "download"

    def download(self, model_id, token_id, save_dir, node_id, filename, format):
        self.node_id = node_id
            
        save_path = self.prepare_download_path(save_dir, filename)
        url = f'https://civitai.com/api/download/models/{model_id}'

        full_filename = filename + format
        
        return self.handle_download(
            DownloadManager.download_with_progress,
            url=url,
            save_path=save_path,
            progress_callback=self,
            params={'token': token_id},
            filename=full_filename
        )
