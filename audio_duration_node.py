import os
import sys
import numpy as np
from typing import Dict, List, Any, Optional
import torchaudio
import torch

# 确保 torchaudio 版本支持所需的音频格式
try:
    import librosa
except ImportError:
    print("警告: 未安装 librosa 库，某些音频格式可能无法处理")
    librosa = None

# 尝试导入 soundfile 以支持更多格式
try:
    import soundfile as sf
except ImportError:
    print("警告: 未安装 soundfile 库，可能无法处理某些音频格式")
    sf = None

class GetAudioDuration:
    """获取音频文件的时长"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_file": ("STRING", {"multiline": False, "default": ""}),
            },
        }
    
    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("duration_seconds", "duration_text")
    FUNCTION = "get_duration"
    CATEGORY = "audio"
    
    def get_duration(self, audio_file: str) -> tuple:
        """获取音频文件的时长并转换为整数秒"""
        try:
            # 检查文件是否存在
            if not os.path.exists(audio_file):
                raise FileNotFoundError(f"文件不存在: {audio_file}")
            
            # 尝试使用 torchaudio 加载音频
            try:
                waveform, sample_rate = torchaudio.load(audio_file)
                duration_seconds = int(waveform.size(1) / sample_rate)
                return (duration_seconds, f"{duration_seconds} 秒")
            except Exception as e:
                print(f"torchaudio 加载失败: {str(e)}", file=sys.stderr)
            
            # 尝试使用 librosa 加载音频
            if librosa is not None:
                try:
                    y, sr = librosa.load(audio_file, sr=None)
                    duration_seconds = int(librosa.get_duration(y=y, sr=sr))
                    return (duration_seconds, f"{duration_seconds} 秒")
                except Exception as e:
                    print(f"librosa 加载失败: {str(e)}", file=sys.stderr)
            
            # 尝试使用 soundfile 加载音频
            if sf is not None:
                try:
                    data, samplerate = sf.read(audio_file)
                    duration_seconds = int(len(data) / samplerate)
                    return (duration_seconds, f"{duration_seconds} 秒")
                except Exception as e:
                    print(f"soundfile 加载失败: {str(e)}", file=sys.stderr)
            
            # 如果所有方法都失败
            raise ValueError(f"无法加载音频文件: {audio_file}")
            
        except Exception as e:
            print(f"获取音频时长时出错: {str(e)}", file=sys.stderr)
            return (0, "错误: " + str(e))

class GetAudioDurationFromAudio:
    """直接从音频张量获取音频时长并转换为整数"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
        }
    
    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("duration_seconds", "duration_text")
    FUNCTION = "get_duration"
    CATEGORY = "audio"
    
    def get_duration(self, audio: Any) -> tuple:
        """从音频张量获取音频时长并转换为整数秒"""
        try:
            print(f"音频类型: {type(audio)}", file=sys.stderr)
            print(f"音频内容: {audio.keys() if isinstance(audio, dict) else audio}", file=sys.stderr)
            
            # 尝试提取ComfyUI标准音频格式
            if isinstance(audio, dict):
                # 检查是否为AudioLoader节点的输出格式
                if "samples" in audio and "sample_rate" in audio:
                    samples = audio["samples"]
                    sample_rate = audio["sample_rate"]
                    
                    if isinstance(samples, torch.Tensor):
                        duration_seconds = int(samples.size(-1) / sample_rate)
                        return (duration_seconds, f"{duration_seconds} 秒")
                    elif isinstance(samples, np.ndarray):
                        duration_seconds = int(samples.shape[-1] / sample_rate)
                        return (duration_seconds, f"{duration_seconds} 秒")
                
                # 检查是否为AudioToAudio节点的输出格式
                if "audio" in audio:
                    return self.get_duration(audio["audio"])
                
                # 检查是否为音频路径
                if "filename" in audio and isinstance(audio["filename"], str):
                    return GetAudioDuration().get_duration(audio["filename"])
                
                # 尝试其他可能的键名
                possible_keys = ["samplerate", "rate", "sr", "audio_data", "waveform"]
                for key in possible_keys:
                    if key in audio:
                        if key in ["samplerate", "rate", "sr"]:
                            sample_rate = audio[key]
                            # 寻找对应的音频数据
                            for data_key in ["data", "samples", "audio", "waveform"]:
                                if data_key in audio:
                                    data = audio[data_key]
                                    if isinstance(data, torch.Tensor):
                                        duration_seconds = int(data.size(-1) / sample_rate)
                                        return (duration_seconds, f"{duration_seconds} 秒")
                                    elif isinstance(data, np.ndarray):
                                        duration_seconds = int(data.shape[-1] / sample_rate)
                                        return (duration_seconds, f"{duration_seconds} 秒")
                        elif key in ["audio_data", "waveform"]:
                            data = audio[key]
                            # 寻找对应的采样率
                            for rate_key in ["samplerate", "rate", "sr", "sample_rate"]:
                                if rate_key in audio:
                                    sample_rate = audio[rate_key]
                                    if isinstance(data, torch.Tensor):
                                        duration_seconds = int(data.size(-1) / sample_rate)
                                        return (duration_seconds, f"{duration_seconds} 秒")
                                    elif isinstance(data, np.ndarray):
                                        duration_seconds = int(data.shape[-1] / sample_rate)
                                        return (duration_seconds, f"{duration_seconds} 秒")
                
                # 尝试更深入地探索字典结构
                for k, v in audio.items():
                    if isinstance(v, dict):
                        result = self.get_duration(v)
                        if result[0] > 0:
                            return result
                    elif isinstance(v, tuple) and len(v) == 2:
                        # 检查是否为 (samples, sample_rate) 元组
                        try:
                            samples, rate = v
                            if isinstance(samples, torch.Tensor):
                                duration_seconds = int(samples.size(-1) / rate)
                                return (duration_seconds, f"{duration_seconds} 秒")
                            elif isinstance(samples, np.ndarray):
                                duration_seconds = int(samples.shape[-1] / rate)
                                return (duration_seconds, f"{duration_seconds} 秒")
                        except:
                            continue
            
            # 如果以上方法都失败，尝试使用 librosa 处理
            if librosa is not None:
                try:
                    # 尝试将音频转换为 numpy 数组
                    if isinstance(audio, torch.Tensor):
                        audio_np = audio.cpu().numpy()
                    elif hasattr(audio, "__array__"):
                        audio_np = np.array(audio)
                    else:
                        # 尝试从字典中提取 numpy 数组
                        for k, v in audio.items():
                            if isinstance(v, torch.Tensor):
                                audio_np = v.cpu().numpy()
                                break
                            elif isinstance(v, np.ndarray):
                                audio_np = v
                                break
                        else:
                            raise ValueError("无法将音频转换为 numpy 数组")
                    
                    # 获取采样率
                    sample_rate = 44100  # 默认采样率
                    for key in ["sample_rate", "samplerate", "rate", "sr"]:
                        if key in audio:
                            sample_rate = audio[key]
                            break
                    
                    # 计算时长
                    if audio_np.ndim > 1:
                        # 如果是多声道，取第一个通道
                        audio_np = audio_np[0]
                    
                    duration_seconds = int(librosa.get_duration(y=audio_np, sr=sample_rate))
                    return (duration_seconds, f"{duration_seconds} 秒")
                except Exception as e:
                    print(f"librosa 处理失败: {str(e)}", file=sys.stderr)
            
            # 如果所有方法都失败
            raise ValueError(f"无法解析音频格式: {type(audio)}, 内容: {list(audio.keys()) if isinstance(audio, dict) else str(audio)[:50]}")
            
        except Exception as e:
            print(f"获取音频时长时出错: {str(e)}", file=sys.stderr)
            return (0, "错误: " + str(e))

# 显示整数值的节点
class DisplayIntegerValue:
    """显示整数值"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
                "prefix": ("STRING", {"default": "计数: ", "multiline": False}),
            },
        }
    
    RETURN_TYPES = ()
    FUNCTION = "display_value"
    OUTPUT_NODE = True
    CATEGORY = "utils"
    
    def display_value(self, value, prefix):
        text = f"{prefix}{value}"
        return {"ui": {"text": [text]}}

# 节点定义
NODE_CLASS_MAPPINGS = {
    "GetAudioDuration": GetAudioDuration,
    "GetAudioDurationFromAudio": GetAudioDurationFromAudio,
    "DisplayIntegerValue": DisplayIntegerValue,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GetAudioDuration": "获取音频时长(文件)",
    "GetAudioDurationFromAudio": "获取音频时长(张量)",
    "DisplayIntegerValue": "显示整数值",
}    