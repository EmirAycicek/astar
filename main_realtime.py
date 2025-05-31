"""
A* Pathfinding Algoritması - Real-time Görselleştirmeli Ana Uygulama
Sakarya Uygulamalı Bilimler Üniversitesi
Algoritma Analizi ve Tasarımı Dersi Ödevi

Bu uygulama A* algoritmasını sıfırdan implement eder ve
real-time görselleştirme ile çalışır.
"""

import sys
import time
import random
from graph import Graph, Node
from astar import AStar, AStarVariant, path_cost, smooth_path
from heuristics import HeuristicSelector, HEURISTIC_INFO
from visualizer import AStarVisualizer, StatisticsVisualizer
from realtime_visualizer import RealtimeAStarVisualizer, StepByStepVisualizer
import matplotlib.pyplot as plt


def safe_input(prompt, default="1"):
    """Güvenli input alma fonksiyonu"""
    try:
        sys.stdout.write(prompt)
        sys.stdout.flush()
        user_input = input().strip()
        return user_input if user_input else default
    except (EOFError, KeyboardInterrupt):
        print(f"\nVarsayılan değer kullanılıyor: {default}")
        return default
    except Exception as e:
        print(f"Input hatası: {e}")
        return default


def create_large_graph(width=50, height=25):
    """Büyük graf oluştur (1000+ düğüm)"""
    print(f"Graf oluşturuluyor: {width}x{height} = {width*height} düğüm")
    
    graph = Graph(width, height)
    graph.create_random_walls(wall_percentage=0.25)
    
    start_x, start_y = 2, 2
    goal_x, goal_y = width - 3, height - 3
    
    graph.remove_wall(start_x, start_y)
    graph.remove_wall(goal_x, goal_y)
    graph.set_start(start_x, start_y)
    graph.set_goal(goal_x, goal_y)
    
    print(f"Başlangıç: ({start_x}, {start_y})")
    print(f"Hedef: ({goal_x}, {goal_y})")
    print(f"Boş düğüm sayısı: {graph.get_empty_nodes_count()}")
    
    return graph


def create_demo_graph(width=30, height=20):
    """Demo için optimal boyut graf"""
    print(f"Demo graf oluşturuluyor: {width}x{height} = {width*height} düğüm")
    
    graph = Graph(width, height)
    graph.create_random_walls(wall_percentage=0.3)
    
    start_x, start_y = 2, 2
    goal_x, goal_y = width - 3, height - 3
    
    graph.remove_wall(start_x, start_y)
    graph.remove_wall(goal_x, goal_y)
    graph.set_start(start_x, start_y)
    graph.set_goal(goal_x, goal_y)
    
    return graph


def create_maze_graph(width=40, height=30):
    """Labirent tarzı graf oluştur"""
    print(f"Labirent oluşturuluyor: {width}x{height} = {width*height} düğüm")
    
    graph = Graph(width, height)
    graph.create_maze_pattern()
    
    for _ in range(100):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        graph.remove_wall(x, y)
    
    graph.set_start(1, 1)
    graph.set_goal(width - 2, height - 2)
    
    return graph


def demonstrate_single_heuristic(graph, heuristic_name='euclidean'):
    """Tek bir heuristic ile örnek çalıştırma"""
    print(f"\n{'='*60}")
    print(f"A* Algoritması - {heuristic_name.upper()} Heuristic")
    print(f"{'='*60}")
    
    astar = AStar(graph, heuristic_name)
    
    start_time = time.time()
    path, success, stats = astar.find_path(step_by_step=True)
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"\nSonuçlar:")
    print(f"• Yol bulundu: {'Evet' if success else 'Hayır'}")
    print(f"• Çalışma süresi: {execution_time:.4f} saniye")
    print(f"• Keşfedilen düğüm: {stats['nodes_explored']}")
    print(f"• Yol uzunluğu: {stats['path_length']}")
    print(f"• Toplam adım: {stats['total_steps']}")
    
    if success:
        print(f"• Yol maliyeti: {path_cost(path, graph)}")
        print(f"• İlk 5 düğüm: {[(n.x, n.y) for n in path[:5]]}")
        print(f"• Son 5 düğüm: {[(n.x, n.y) for n in path[-5:]]}")
    
    return astar, path, success, stats


def real_time_demo():
    """Real-time A* görselleştirme demo"""
    print(f"\n{'='*60}")
    print("🎬 REAL-TIME A* GÖRSELLEŞTİRME")
    print(f"{'='*60}")
    
    # Graf seçenekleri
    print("\nGraf tipi seçin:")
    print("1. Demo Graf (30x20 - 600 düğüm)")
    print("2. Orta Graf (40x25 - 1000 düğüm)")  
    print("3. Büyük Graf (50x30 - 1500 düğüm)")
    print("4. Labirent (35x25 - 875 düğüm)")
    
    graf_choice = safe_input("Graf seçimi (1-4): ", "1")
    
    if graf_choice == "2":
        graph = create_large_graph(40, 25)
    elif graf_choice == "3":
        graph = create_large_graph(50, 30)
    elif graf_choice == "4":
        graph = create_maze_graph(35, 25)
    else:
        graph = create_demo_graph(30, 20)
    
    # Heuristic seçimi
    selector = HeuristicSelector()
    heuristics = selector.get_all_names()
    
    print(f"\nHeuristic seçin:")
    for i, h in enumerate(heuristics[:5], 1):  # İlk 5 tanesini göster
        info = HEURISTIC_INFO.get(h, {})
        name = info.get('name', h)
        print(f"{i}. {name}")
    
    h_choice = safe_input(f"Heuristic seçimi (1-5): ", "2")
    
    try:
        h_index = int(h_choice) - 1
        if 0 <= h_index < len(heuristics):
            selected_heuristic = heuristics[h_index]
        else:
            selected_heuristic = 'euclidean'
    except ValueError:
        selected_heuristic = 'euclidean'
    
    print(f"\nSeçilen heuristic: {selected_heuristic}")
    
    # Görselleştirme modu
    print(f"\nGörselleştirme modu:")
    print("1. Otomatik - Yavaş (0.3s arası)")
    print("2. Otomatik - Normal (0.15s arası)")
    print("3. Otomatik - Hızlı (0.05s arası)")
    print("4. Manuel - Adım adım (ENTER ile)")
    
    viz_choice = safe_input("Mod seçimi (1-4): ", "2")
    
    # A* ve visualizer oluştur
    astar = AStar(graph, selected_heuristic)
    step_visualizer = StepByStepVisualizer(graph, astar)
    
    try:
        print(f"\n🚀 Real-time görselleştirme başlatılıyor...")
        print(f"💡 İpucu: Görselleştirme sırasında pencereyi kapatmayın!")
        
        time.sleep(1)  # Kısa bekleme
        
        if viz_choice == "1":
            result = step_visualizer.run_auto_speed("slow")
        elif viz_choice == "3":
            result = step_visualizer.run_auto_speed("fast")
        elif viz_choice == "4":
            result = step_visualizer.run_step_by_step()
        else:
            result = step_visualizer.run_auto_speed("normal")
        
        if result:
            print("\n✅ Real-time görselleştirme tamamlandı!")
            print("📊 Pencereyi açık bırakabilir veya kapatabilirsiniz.")
            
            save_choice = safe_input("\nSonucu PNG olarak kaydetmek ister misiniz? (e/h): ", "h")
            if save_choice.lower() == 'e':
                plt.savefig(f'realtime_result_{selected_heuristic}.png', dpi=300, bbox_inches='tight')
                print("✅ Sonuç kaydedildi!")
    
    except Exception as e:
        print(f"\n💥 Real-time görselleştirme hatası: {e}")
        print("Matplotlib veya görüntü sistemi problemi olabilir.")


def compare_all_heuristics(graph):
    """Tüm heuristic'leri karşılaştır"""
    print(f"\n{'='*60}")
    print("TÜM HEURİSTİC'LERİ KARŞILAŞTIRMA")
    print(f"{'='*60}")
    
    heuristic_selector = HeuristicSelector()
    heuristics = heuristic_selector.get_all_names()
    
    results = {}
    
    for heuristic in heuristics:
        print(f"\n{heuristic} test ediliyor...")
        
        astar = AStar(graph, heuristic)
        start_time = time.time()
        path, success, stats = astar.find_path(step_by_step=False)
        end_time = time.time()
        
        results[heuristic] = {
            'path': path,
            'success': success,
            'stats': stats,
            'execution_time': end_time - start_time
        }
        
        status = "✓" if success else "✗"
        print(f"  {status} {heuristic}: {stats['nodes_explored']} düğüm keşfedildi")
    
    StatisticsVisualizer.create_performance_table(results)
    return results


def interactive_demo():
    """İnteraktif demo"""
    print(f"\n{'='*60}")
    print("İNTERAKTİF A* DEMOsu")
    print(f"{'='*60}")
    
    # Graf boyutunu kullanıcıdan al
    try:
        width_input = safe_input("Graf genişliği (varsayılan 30): ", "30")
        height_input = safe_input("Graf yüksekliği (varsayılan 20): ", "20")
        
        width = int(width_input) if width_input.isdigit() else 30
        height = int(height_input) if height_input.isdigit() else 20
    except ValueError:
        print("Geçersiz boyut! Varsayılan boyut kullanılıyor.")
        width, height = 30, 20
    
    # Graf tipini seç
    print("\nGraf tipi seçin:")
    print("1. Rastgele duvarlar")
    print("2. Labirent deseni")
    
    choice = safe_input("Seçiminiz (1 veya 2): ", "1")
    
    if choice == "2":
        graph = create_maze_graph(width, height)
    else:
        graph = create_demo_graph(width, height)
    
    # Heuristic seç
    selector = HeuristicSelector()
    heuristics = selector.get_all_names()
    
    print(f"\nMevcut heuristic fonksiyonları:")
    for i, h in enumerate(heuristics[:6], 1):
        info = HEURISTIC_INFO.get(h, {})
        name = info.get('name', h)
        desc = info.get('description', 'Açıklama yok')
        print(f"{i}. {name}: {desc}")
    
    h_choice_input = safe_input(f"\nHeuristic seçimi (1-6): ", "2")
    
    try:
        h_choice = int(h_choice_input) - 1
        if 0 <= h_choice < min(6, len(heuristics)):
            selected_heuristic = heuristics[h_choice]
        else:
            selected_heuristic = 'euclidean'
    except ValueError:
        print("Geçersiz seçim! Euclidean kullanılıyor.")
        selected_heuristic = 'euclidean'
    
    print(f"\nSeçilen heuristic: {selected_heuristic}")
    
    # Görselleştirme seçenekleri
    print(f"\nGörselleştirme seçenekleri:")
    print("1. Real-time Canlı Görselleştirme (ÖNERİLİR)")
    print("2. Sadece sonucu göster")
    print("3. Animasyon oluştur (GIF)")
    print("4. Tüm heuristic'leri karşılaştır")
    
    vis_choice = safe_input("Seçiminiz (1-4): ", "1")
    
    try:
        if vis_choice == "1":
            # Real-time görselleştirme
            astar = AStar(graph, selected_heuristic)
            step_visualizer = StepByStepVisualizer(graph, astar)
            
            speed_choice = safe_input("Hız (1:Yavaş, 2:Normal, 3:Hızlı): ", "2")
            speeds = {"1": "slow", "2": "normal", "3": "fast"}
            speed = speeds.get(speed_choice, "normal")
            
            step_visualizer.run_auto_speed(speed)
            
        elif vis_choice == "2":
            astar, path, success, stats = demonstrate_single_heuristic(graph, selected_heuristic)
            visualizer = AStarVisualizer(graph, astar)
            fig = visualizer.show_final_result(path, stats)
            visualizer.show()
            
        elif vis_choice == "3":
            from visualizer import AStarVisualizer
            astar = AStar(graph, selected_heuristic)
            visualizer = AStarVisualizer(graph, astar)
            
            path, success, stats = astar.find_path(step_by_step=True)
            if success:
                animation_obj = visualizer.animate_algorithm(
                    interval=150, save_gif=True,
                    filename=f'astar_{selected_heuristic}_{width}x{height}.gif'
                )
                print("✅ GIF animasyonu oluşturuldu!")
            
        elif vis_choice == "4":
            results = compare_all_heuristics(graph)
            astar = AStar(graph, 'euclidean')  # Karşılaştırma için
            visualizer = AStarVisualizer(graph, astar)
            fig, _ = visualizer.compare_heuristics_visualization(list(results.keys())[:6])
            plt.show()
            
    except Exception as e:
        print(f"Görselleştirme hatası: {e}")
        print("Matplotlib kurulu değil veya display problemi var.")


def run_performance_tests():
    """Performans testleri"""
    print(f"\n{'='*60}")
    print("PERFORMANS TESTLERİ")
    print(f"{'='*60}")
    
    test_sizes = [(20, 15), (30, 20), (40, 25), (50, 30)]
    
    for width, height in test_sizes:
        print(f"\n{width}x{height} graf test ediliyor...")
        
        graph = create_large_graph(width, height)
        astar = AStar(graph, 'euclidean')
        
        start_time = time.time()
        path, success, stats = astar.find_path(step_by_step=False)
        end_time = time.time()
        
        print(f"  Sonuç: {'Başarılı' if success else 'Başarısız'}")
        print(f"  Süre: {end_time - start_time:.4f} saniye")
        print(f"  Keşfedilen: {stats['nodes_explored']} düğüm")
        print(f"  Yol uzunluğu: {stats['path_length']}")


def main():
    """Ana fonksiyon"""
    print("=" * 70)
    print("🎯 A* PATHFINDING ALGORİTMASI - REAL-TIME VERSİYON")
    print("Sakarya Uygulamalı Bilimler Üniversitesi")
    print("Algoritma Analizi ve Tasarımı Dersi")
    print("=" * 70)
    
    print("\nÇalışma modu seçin:")
    print("1. 🎬 Real-time Canlı Görselleştirme (ÖNERİLİR)")
    print("2. 🚀 Hızlı demo")
    print("3. 🎮 İnteraktif demo")
    print("4. ⚡ Performans testleri")
    print("5. 📊 Tüm heuristic karşılaştırması")
    print("\n💡 Not: Real-time görselleştirme algoritmanın adım adım çalışmasını gösterir!")
    
    choice = safe_input("\nSeçiminiz (1-5): ", "1")
    print(f"Seçilen: {choice}")
    
    try:
        if choice == "1":
            # Real-time görselleştirme
            real_time_demo()
            
        elif choice == "2":
            # Hızlı demo
            print("\n🚀 Hızlı demo başlatılıyor...")
            graph = create_large_graph(50, 30)
            astar, path, success, stats = demonstrate_single_heuristic(graph, 'euclidean')
            
            if success:
                print("\n📊 Görselleştirme oluşturuluyor...")
                try:
                    visualizer = AStarVisualizer(graph, astar)
                    fig = visualizer.show_final_result(path, stats)
                    visualizer.save_image('astar_demo_result.png')
                    print("✅ Sonuç 'astar_demo_result.png' olarak kaydedildi!")
                except Exception as e:
                    print(f"Görselleştirme hatası: {e}")
            
        elif choice == "3":
            interactive_demo()
            
        elif choice == "4":
            run_performance_tests()
            
        elif choice == "5":
            graph = create_large_graph(40, 30)
            results = compare_all_heuristics(graph)
            
            try:
                fig = StatisticsVisualizer.plot_performance_comparison(results)
                plt.savefig('heuristic_comparison.png', dpi=300, bbox_inches='tight')
                print("✅ Karşılaştırma grafiği 'heuristic_comparison.png' olarak kaydedildi!")
            except Exception as e:
                print(f"Grafik oluşturma hatası: {e}")
            
        else:
            print("🔄 Geçersiz seçim! Real-time demo çalıştırılıyor...")
            real_time_demo()
            
    except KeyboardInterrupt:
        print("\n\n❌ Program kullanıcı tarafından durduruldu.")
    except Exception as e:
        print(f"\n💥 Hata oluştu: {e}")
        print("\n📦 Lütfen gerekli kütüphanelerin yüklü olduğundan emin olun:")
        print("pip install matplotlib numpy pillow")


if __name__ == "__main__":
    # Platform ve encoding ayarları
    try:
        import sys
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        if hasattr(sys.stdin, 'reconfigure'):
            sys.stdin.reconfigure(encoding='utf-8')
    except:
        pass
    
    main()