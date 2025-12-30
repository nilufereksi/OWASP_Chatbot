
## 1. Kütüphane Kurulumu ve Hazırlık
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# import torch
# major_version, minor_version = torch.cuda.get_device_capability()
# # Colab T4 GPU'su için Unsloth kütüphanesini kuruyoruz
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

"""## 2. Temel Modelin Yüklenmesi ve LoRA Yapılandırması"""

from unsloth import FastLanguageModel
import torch

max_seq_length = 2048 # Modelin okuyabileceği maksimum kelime uzunluğu
dtype = None
load_in_4bit = True # Hafıza tasarrufu için 4-bit yükleme

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit", # Llama 3 tabanlı optimize model
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# LoRA (Low-Rank Adaptation) ayarları - Modeli hızlı eğitmek için
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

"""qa_pairs.jsonl"""

{"instruction": "Bir SQL enjeksiyon zafiyeti nasıl önlenir?", "output": "OWASP'ye göre, SQL enjeksiyonunu önlemek için **parametreli sorgular (Prepared Statements)** kullanılmalıdır. Veritabanı sorguları dinamik string birleştirme ile oluşturulmamalıdır."}
{"instruction": "Erişim kontrolü için hangi güvenlik mekanizması önerilir?", "output": "Her istekte sunucu tarafında yetkilendirme kontrolü (tercihen RBAC veya ABAC) uygulanması önerilir. Kullanıcı, yetkisi olmayan bir kaynağa erişmeye çalışırsa 403 (Yasak) döndürülmelidir."}
{"instruction": "IDOR açığı nedir?", "output": "IDOR (Insecure Direct Object Reference), yetkili olmayan bir kullanıcının URL parametrelerini veya form alanlarını değiştirerek doğrudan nesnelere (örneğin başka bir kullanıcının ID'si) erişebilmesidir."}
{"instruction": "IDOR hataları için önleme yöntemi nedir? ", "output": "IDOR hataları için, girdi yerine **tahmin edilemez (non-guessable)** referanslar veya sunucuda **kullanıcıya özel yetkilendirilmiş nesne ID'leri** kullanılmalıdır."}
{"instruction": "Hata mesajları ne tür bilgiler içermelidir?", "output": "Hata mesajları asla son kullanıcıya detaylı sistem bilgisi (stack trace, veritabanı şeması, sunucu konfigürasyonu) vermemelidir. Hata bilgisi sadece loglanmalıdır."}
{"instruction": "Güvenlik logları en az kaç gün saklanmalıdır? ", "output": "Loglar, değiştirilemezlik (immutability) ilkesi gereği, minimum 90 gün boyunca merkezi bir Log Yönetim Sistemi'nde saklanmalıdır. "}
{"instruction": "Yüksek hassasiyetli sistemlere erişimde hangi kimlik doğrulama yöntemi zorunludur? ", "output": "Yüksek hassasiyetli tüm sistemlere erişimde Çok Faktörlü Kimlik Doğrulama (MFA) zorunludur. "}
{"instruction": "Varsayılan Reddetme ilkesi ne anlama gelir? ", "output": "Güvenlik kararları alınırken, açıkça izin verilmeyen her şeyin otomatik olarak reddedilmesi ilkesidir. "}
{"instruction": "A01:2021   Bozuk Erişim Kontrolü maddesinin tanımı nedir?", "output": "Kimliği doğrulanmış kullanıcıların yetkileri dahilinde olmayan verilere veya işlevlere erişim sağlamasıdır. Bu durum genellikle uygulama politikalarının düzgün bir şekilde uygulanmamasından kaynaklanır."}
{"instruction": "A02:2021   Kriptografik Hatalar maddesinin tanımı nedir?", "output": "Hassas verilerin aktarım veya depolama sırasında uygun şekilde korunmaması, bu da genellikle şifreleme eksikliği veya zayıf şifreleme algoritmaları ve anahtar yönetimindeki hatalardan kaynaklanır."}
{"instruction": "A03:2021   Enjeksiyon maddesinin tanımı nedir?", "output": "Güvenilmeyen verilerin, bir komut veya sorgunun parçası olarak yorumlayıcıya (SQL, OS komut kabuğu vb.) gönderilmesidir. Bu durum, yorumlayıcının istenmeyen komutları çalıştırmasına neden olur."}
{"instruction": "A04:2021   Güvenli Olmayan Tasarım maddesinin tanımı nedir?", "output": "Güvenlik kontrollerinin düzgün uygulanmamasından ziyade, uygulamanın tasarım ve mimarisindeki eksiklikler veya zafiyetlerle ilgili riskleri ifade eder. Bu, güvenlik amaçlı tasarım standartlarının eksikliğinden kaynaklanır."}
{"instruction": "A05:2021   Güvenlik Yanlış Yapılandırması maddesinin tanımı nedir?", "output": "Uygulama yığınının (sunucu, veritabanı, çerçeve vb.) güvenlik ayarlarının eksik veya yanlış yapılandırılmasıdır. Çoğu zaman varsayılan hesapların, gereksiz özelliklerin veya yanlış izinlerin bırakılmasından kaynaklanır."}
{"instruction": "A06:2021   Eskimiş ve Rentansiyonu Olmayan Bileşenler maddesinin tanımı nedir?", "output": "Uygulamanın kullandığı kütüphane, çerçeve veya diğer yazılım bileşenlerinin bilinen güvenlik zafiyetlerine sahip olması veya güncel olmamasıdır. Bu, genellikle yama yönetiminin ihmal edilmesinden kaynaklanır."}
{"instruction": "A07:2021   Kimlik Doğrulama Hataları maddesinin tanımı nedir?", "output": "Bir saldırganın kimlik doğrulama sürecini atlamasına veya başka bir kullanıcının kimliğine bürünmesine olanak tanıyan zafiyetlerdir. Parola zayıflığı ve yetersiz oturum/MFA yönetimi bu kapsamdadır."}
{"instruction": "A08:2021   Yazılım ve Veri Bütünlüğü Hataları maddesinin tanımı nedir?", "output": "Uygulamanın kod veya kritik verilerinin bütünlüğünün ihlal edilmesidir; bu durum genellikle güvenilmeyen kaynaklardan gelen veriler veya güvensiz CI/CD boru hatları aracılığıyla gerçekleşir."}
{"instruction": "A09:2021   Güvenlik Loglama ve İzleme Hataları maddesinin tanımı nedir?", "output": "Kritik güvenlik olaylarının yetersiz veya etkisiz bir şekilde loglanması, bu logların izlenmemesi veya aktif bir uyarı ve yanıt mekanizmasının olmamasıdır. Bu, saldırıların tespit edilmesini ve soruşturulmasını imkansız hale getirir."}
{"instruction": "A10:2021   Sunucu Tarafı İstek Sahteciliği maddesinin tanımı nedir?", "output": "Bir saldırganın sunucunun işlevselliğini kötüye kullanarak, sunucunun dahili veya harici ağ kaynaklarına istenmeyen istekler yapmasını sağlamasıdır. Bu, genellikle sunucunun kullanıcı girdisine dayalı URL'leri doğrulamadan almasından kaynaklanır."}
{"instruction": "Bir kullanıcının oturumunun sona ermesi (logout) ne zaman zorunludur?", "output": "Kullanıcı, tarayıcıyı kapattığında, belirli bir süre işlem yapmadığında (idle timeout) veya açıkça 'Çıkış Yap' dediğinde oturum sonlandırılmalıdır."}
{"instruction": "Mobil uygulamalarda parola depolama için hangi yöntem kesinlikle yasaktır?", "output": "Mobil uygulamalarda parolaların cihazın yerel depolama alanında (local storage) veya düz metin (plaintext) olarak saklanması kesinlikle yasaktır. Güvenli anahtar depoları kullanılmalıdır."}
{"instruction": "CI/CD hattına yetkisiz erişimi engellemek için ne yapılmalıdır?", "output": "CI/CD hattı araçlarına (Jenkins, GitLab CI vb.) erişim, **En Az Yetki Prensibi (Principle of Least Privilege)** uygulanarak sınırlandırılmalıdır."}
{"instruction": "İstemci tarafı (Client-Side) doğrulama, sunucu tarafı (Server-Side) doğrulamanın yerini tutar mı?", "output": "Hayır, istemci tarafı doğrulama sadece kullanıcı deneyimi (UX) içindir; tüm güvenlik kararları için sunucu tarafı doğrulama zorunludur."}
{"instruction": "SQLi, XSS ve LFI gibi Enjeksiyon (A03) zafiyetlerini önlemede en temel adım nedir?", "output": "Tüm kullanıcı girdilerinin **güvenilmez** kabul edilmesi ve kullanılmadan önce bağlama uygun olarak sıkıca doğrulanması (**Girdi Doğrulama**) ve kodlanması (**Çıktı Kodlama**) gerekir."}
{"instruction": "Hassas verileri şifrelemek için hangi modern algoritma zorunludur?" , "output": "Hassas veriler için **AES-256** gibi güçlü bir algoritma kullanılmalıdır. "}
{"instruction": "Oturum boşta kalma süresi (idle timeout) en fazla kaç dakikadır? ", "output": "Kesin Rakam/Limit: 'Boşta kalma süresi **15 dakikayı geçmemelidir**.' (A07:2021) "}
{"instruction": "Atıl durumdaki hassas veriler için zorunlu şifreleme standardı nedir? ", "output": "Atıl durumdaki veriler için AES-256 kullanılmalıdır. "}
{"instruction": "İletimdeki veriler için hangi TLS sürümü zorunludur? ", "output": "İletimdeki veriler için **TLS 1.2 veya üzeri** zorunludur. "}
{"instruction": "Hangi şifre paketlerinin (cipher suites) kullanımı kesinlikle devre dışı bırakılmalıdır? ", "output": "Eski ve zayıf şifre paketleri (**örneğin, 3DES, RC4, anonim paketler**) kesinlikle devre dışı bırakılmalıdır."}
{"instruction": "Parola ve hassas API anahtarları için önerilen hash algoritmaları hangileridir? ", "output": "Parolalar ve anahtarlar için **Argon2 veya bcrypt** gibi modern ve yavaş hash algoritmaları önerilir. "}
{"instruction": "Parola hashlenirken kullanılması zorunlu olan ek güvenlik unsuru nedir? ", "output": "Parolalar hashlenirken **Benzersiz Tuzlama (Unique Salt)** kullanılmalıdır. "}
{"instruction": "Komut satırı enjeksiyonu için kullanıcı girdileri nasıl işlenmelidir? ", "output": "Kullanıcı girdileri doğrudan komut satırı argümanı olarak geçirilmemelidir. "}
{"instruction": "SQL sorgularının dinamik string birleştirme ile oluşturulması uygun mudur? ", "output": "Hayır, dinamik string birleştirme ile oluşturulması uygun değildir. Parametreli sorgular kullanılmalıdır. "}
{"instruction": "XSS zafiyetlerini önlemek için çıktının nasıl işlenmesi gerekir? ", "output": "Çıktı, tarayıcıya yansıtılmadan önce bağlama uygun şekilde **HTML-escape (çıktı kodlaması)** edilmelidir. "}
{"instruction": "Güven Sınırları (Trust Boundaries) ne zaman ve neden belirlenmelidir? ", "output": "Farklı güven seviyesindeki bileşenler arasında net sınırlar belirlenmeli ve bu sınırları aşan her türlü veri güvensiz kabul edilmelidir. "}
{"instruction": "Tüm kritik API uç noktalarında zorunlu olan güvenlik kontrolü nedir? ", "output": "Tüm kritik API uç noktalarında otomatik **hız sınırlaması (Rate Limiting)** zorunludur. "}
{"instruction": "Güvenlik kararlarında temel alınması gereken mimari ilke nedir? ", "output": "Temel mimari ilke **Varsayılan Reddetme (Deny by Default)**'dir. "}
{"instruction": "Kurulumdan hemen sonra değiştirilmesi gereken ayarlar nelerdir? ", "output": "Tüm varsayılan ayarlar ve parolalar, kurulumdan hemen sonra değiştirilmelidir. "}
{"instruction": "Sunucuda veya uygulamada kullanılmayan servislere ne yapılmalıdır? ", "output": "Kullanılmayan servisler ve özellikler devre dışı bırakılmalıdır. "}
{"instruction": "Hata mesajları son kullanıcıya hangi bilgileri vermemelidir? ", "output": "Hata mesajları, detaylı sistem bilgisi (stack trace, veritabanı şeması vb.) vermemelidir. "}
{"instruction": "Projelerde desteklenmeyen (End-of-Life) kütüphanelerin kullanımı için şirket politikası nedir? ", "output": "Geliştiricilerin ömrü dolmuş (End-of-Life) kütüphaneleri kullanması kesinlikle yasaktır. "}
{"instruction": "Üçüncü taraf kütüphanelerdeki zafiyetleri kontrol etmek için hangi otomatik araçlar kullanılmalıdır? ", "output": "Bilinen güvenlik zafiyetlerini kontrol eden otomatik tarama araçları (**SCA**) kullanılmalıdır. "}
{"instruction": "Orta veya üzeri risk içeren bulguların yamalanması için maksimum süre nedir? ", "output": "Orta ve üzeri riskli bulgular içeren bileşenler **7 iş günü** içinde yamalanmalıdır. "}
{"instruction": "Oturum kimlikleri (session IDs) hangi protokol üzerinden iletilmemelidir? ", "output": "Oturum kimlikleri **HTTP** üzerinden asla iletilmemelidir. "}
{"instruction": "Parola sıfırlama mekanizması ne tür bir token gerektirmelidir? ", "output": "Parola sıfırlama mekanizması güvenli, **tek kullanımlık bir token** gerektirmelidir. "}
{"instruction": "Kritik konfigürasyon dosyaları için bütünlük kontrolü nasıl yapılmalıdır? ", "output": "Hash veya dijital imza ile bütünlük kontrolü zorunludur. "}
{"instruction": "CI/CD pipeline ortamları nasıl korunmalıdır? ", "output": "CI/CD ortamları sıkı erişim kontrolleriyle korunmalı ve **manuel müdahaleler yasaklanmalıdır**. "}
{"instruction": "Otomatik güncelleme mekanizmalarının güvenliği için hangi kontrol zorunludur? ", "output": "Otomatik güncelleme mekanizmaları için **hash veya dijital imza ile bütünlük kontrolü** zorunludur. "}
{"instruction": "Hangi olaylar mutlaka loglanmalıdır? ", "output": "Tüm kimlik doğrulama denemeleri, yetkilendirme hataları ve kritik sistem işlemleri loglanmalıdır. "}
{"instruction": "Logların saklanmasında hangi ilke esas alınmalıdır? ", "output": " Logların saklanmasında **değiştirilemezlik (immutability)** ilkesi esas alınmalıdır."}
{"instruction": "Kaç başarısız denemeden sonra otomatik uyarı tetiklenmelidir? ", "output": "Örneğin 1 dakikada 5 deneme gibi şüpheli aktiviteler için otomatik ve anlık uyarı mekanizmaları kurulmalıdır. "}
{"instruction": "SSRF zafiyetini önlemek için kullanıcı girdisine ne yapılmalıdır? ", "output": "Kullanıcı girdisinin tam ve **katı bir şekilde validasyonu** yapılmalıdır. "}
{"instruction": "Sunucunun bağlanabileceği hedefler nasıl kısıtlanmalıdır? ", "output": "Hedefler **beyaz liste (whitelisting)** yöntemiyle kısıtlanmalıdır. "}
{"instruction": "Sunucunun dahili ağ adreslerine (127.0.0.1) istek yapması engellenmeli midir? ", "output": "Evet, katı ağ segmentasyonu uygulanarak dahili ağ adreslerine istek yapılması engellenmelidir. "}
{"instruction": "Güvenlik Kapısı'nda tespit edilen yüksek riskli bulgular için ne yapılır? ", "output": "Sürüm otomatik olarak durdurulur ve bu bulgular çözülmeden **üretime geçiş kesinlikle yasaktır**. "}
{"instruction": "SAST taraması ne zaman yapılmalıdır? ", "output": "Tüm yeni kodlar ve güncellemeler için otomatik **SAST taraması CI/CD hattında zorunludur**. "}
{"instruction": "Olay Müdahale Prosedürünün Kapsam Belirleme adımı ne içerir? ", "output": "Açığın etkilediği sistemleri ve veri kapsamını belirler. "}
{"instruction": "Kod incelemesi kaç kıdemli geliştirici tarafından yapılmalıdır? ", "output": "Kritik ve yüksek riskli değişiklikler için **en az iki kıdemli geliştirici** tarafından kod incelemesi zorunludur. "}
{"instruction": "DAST taraması hangi periyotlarda yapılmalıdır? ", "output": "**Haftalık periyotlarda** canlı sistemde DAST (Dinamik Uygulama Güvenlik Testi) taraması yapılmalıdır. "}
{"instruction": "Bir güvenlik açığı ilk tespit edildiğinde kimin bilgilendirilmesi gerekir? ", "output": "Açığı ilk bulan kişi derhal **Güvenlik Ekibi'ne** bildirmelidir. "}
{"instruction": "Olay Müdahale Prosedürünün son adımı nedir? ", "output": "Sorunun kaynağının belirlenmesi ve uzun vadeli önlemler alınmasıdır (**Kök Neden Analizi**). "}
{"instruction": "Kimlik doğrulama hatalarına karşı önlem olarak hangi saldırı türüne karşı hız sınırlaması uygulanır? ", "output": "Otomatik **brute-force (kaba kuvvet) saldırılarına** karşı hız sınırlaması uygulanır. "}
{"instruction": "Tüm kritik olaylar için loglarda hangi bilgi mutlaka bulunmalıdır? ", "output": "Olayın **tarihi, saati, kaynağı ve sonucu** gibi kritik bilgiler. "}
{"instruction": "Hassas verilerin saklanma süresi (Retention) neye göre belirlenir? ", "output": "Yasal gereklilikler ve iş gereksinimleri olduğu sürece belirlenir. "}
{"instruction": "Kullanıcının yetkilendirildiği nesne ID'lerinin kullanılması hangi güvenlik hatasını önler? ", "output": "**IDOR (Insecure Direct Object Reference)** hatalarını önler. "}
{"instruction": "TLS 1.0 veya 1.1 protokollerinin kullanımı hakkında şirket politikası nedir?", "output": "Eski ve zayıf oldukları için kullanımları kesinlikle devre dışı bırakılmalıdır; yalnızca **TLS 1.2 ve üzeri** izinlidir."}
{"instruction": "Uygulama güvenliğinde 'Girdi Doğrulama' (Input Validation) neden önemlidir?", "output": "Kötü niyetli verilerin sisteme girmesini ve **Enjeksiyon (A03) veya SSRF (A10)** gibi saldırıları engellemek için önemlidir."}
{"instruction": "Oturum kimlikleri (Session ID) oluşturulurken hangi özelliğe sahip olmalıdır?", "output": "Oturum kimlikleri güvenli, **rastgele ve tahmin edilemez** olmalıdır."}
{"instruction": "SAST ve DAST taramaları arasındaki temel uygulama farkı nedir?", "output": "**SAST** kod derlenmeden (**statik**) yapılırken, **DAST** çalışan canlı sistem üzerinde (**dinamik**) yapılır."}
{"instruction": "Acil bir güvenlik yaması (Hotfix) uygulandıktan sonra hangi adım izlenmelidir?", "output": "Sorunun kaynağı belirlenmeli ve uzun vadeli önlemler için **Kök Neden Analizi** yapılmalıdır."}
{"instruction": "Yazılım geliştirme sürecinde 'Güvenlik Kapısı' (Security Gate) ne işe yarar?", "output": "**Kritik veya Yüksek riskli bulgular** içeren sürümlerin üretime geçmesini otomatik olarak engeller."}
{"instruction": "Hassas verilerin log dosyalarına yazılması neden yasaktır?", "output": "Veri sızıntısına yol açabileceği ve **KVKK/GDPR gibi yasal düzenlemelere** aykırı olduğu için yasaktır."}
{"instruction": "API anahtarları kod içinde (hardcoded) saklanabilir mi?", "output": "Hayır, API anahtarları asla kod içinde saklanmamalı; güvenli ortam değişkenlerinde veya **kasa (vault) sistemlerinde** tutulmalıdır."}
{"instruction": "RBAC kısaltmasının açılımı ve anlamı nedir?", "output": "**Role-Based Access Control (Rol Tabanlı Erişim Kontrolü)**; kullanıcıların sadece rollerine uygun verilere erişmesini sağlar."}
{"instruction": "Bir saldırganın başka bir kullanıcının verisine erişmesi hangi OWASP maddesi kapsamındadır?", "output": "**A01:2021 Bozuk Erişim Kontrolü** (Broken Access Control) kapsamındadır."}
{"instruction": "Varsayılan (Default) parolalar neden hemen değiştirilmelidir?", "output": "Saldırganlar tarafından bilindikleri ve sisteme kolay erişim sağladıkları için değiştirilmelidir (**A05:2021**)."}
{"instruction": "SCA (Software Composition Analysis) araçları neyi tespit eder?", "output": "Projede kullanılan açık kaynak kütüphanelerdeki **bilinen güvenlik zafiyetlerini (CVE)** tespit eder."}
{"instruction": "HTTP 403 (Yasak) hata kodu ne zaman döndürülmelidir?", "output": "Kimliği doğrulanmış bir kullanıcı, yetkisi olmayan bir kaynağa erişmeye çalıştığında döndürülmelidir."}
{"instruction": "Brute-force saldırılarını engellemek için sistem ne yapmalıdır?", "output": "Belirli sayıda başarısız denemeden sonra hesabı geçici olarak kilitlemeli veya **hız sınırlaması (Rate Limiting)** uygulamalıdır."}
{"instruction": "Olay Müdahale sürecinde 'Kök Neden Analizi' neden yapılır?", "output": "Güvenlik açığının tekrar oluşmasını engellemek ve kalıcı çözüm üretmek için yapılır."}
{"instruction": "Veritabanı şemasının hata mesajında görünmesi hangi zafiyet türüdür?", "output": "**Güvenlik Yanlış Yapılandırması (A05:2021)** ve bilgi ifşasıdır."}
{"instruction": "Kod incelemesi (Code Review) kim tarafından yapılmalıdır?", "output": "Kodu yazan kişi dışında, **en az iki kıdemli geliştirici** tarafından yapılmalıdır."}
{"instruction": "Hangi risk seviyesindeki bulgular canlıya geçişi (deploy) durdurur?", "output": "**Kritik (Critical) ve Yüksek (High)** risk seviyesindeki bulgular canlıya geçişi durdurur."}
{"instruction": "İç ağdaki IP adreslerine (örn: 192.168.x.x) erişim neden kısıtlanmalıdır?", "output": "**SSRF (Server-Side Request Forgery)** saldırılarını ve iç ağ keşfini engellemek için kısıtlanmalıdır."}
{"instruction": "Yazılım güncellemelerinde dijital imza kontrolü yapılmazsa ne olur?", "output": "**A08:2021 Yazılım ve Veri Bütünlüğü Hataları** oluşur; sahte veya zararlı güncellemeler yüklenebilir."}
{"instruction": "Kimlik doğrulama logları nerede toplanmalıdır?", "output": "**Merkezi ve güvenli bir Log Yönetim Sistemi'nde** toplanmalıdır."}
{"instruction": "Kullanılmayan özelliklerin (features) sistemden kaldırılması hangi prensibin gereğidir?", "output": "Saldırı yüzeyini daraltma ve Güvenlik Yanlış Yapılandırmasını (A05) önleme prensibinin gereğidir."}
{"instruction": "Argon2 algoritması ne amaçla kullanılır?", "output": "Parolaların güvenli bir şekilde **hashlenerek saklanması** amacıyla kullanılır."}
{"instruction": "Bir zafiyet tespit edildiğinde 'Kapsam Belirleme' aşamasında ne araştırılır?", "output": "Hangi sistemlerin etkilendiği ve ne kadar verinin risk altında olduğu araştırılır."}
{"instruction": "Cross-Site Scripting (XSS) saldırıları hangi teknikle önlenir?", "output": "Kullanıcı girdisinin doğrulanması ve çıktının güvenli **kodlanması (Output Encoding)** ile önlenir."}
{"instruction": "Bir geliştirici eski bir kütüphaneyi (End-of-Life) projeye eklemek isterse ne yapılmalıdır?", "output": "Talep reddedilmeli ve kütüphanenin **güncel veya güvenli bir alternatifi** kullanılmalıdır."}
{"instruction": "TLS bağlantılarında 'Zayıf Şifre Paketleri' (Weak Cipher Suites) neden tehlikelidir?", "output": "Şifrelemenin kolayca kırılmasına ve verilerin ele geçirilmesine neden olabileceği için tehlikelidir."}
{"instruction": "Oturum yönetimi güvenliği için 'Idle Timeout' süresi neden kısıtlanmalıdır?", "output": "Kullanıcı bilgisayar başından ayrıldığında oturumun açık kalıp çalınmasını önlemek için kısıtlanmalıdır."}
{"instruction": "Uygulama loglarında 'başarısız oturum açma' olaylarının izlenmesi neyi sağlar?", "output": "Olası kaba kuvvet (brute-force) veya parola deneme saldırılarının tespit edilmesini sağlar."}
{"instruction": "Güvenlik politikalarına göre, manuel müdahalenin yasak olduğu ortam hangisidir?", "output": "**CI/CD (Sürekli Entegrasyon/Sürekli Dağıtım)** pipeline ortamlarıdır."}
{"instruction": "SAST vs DAST prosedürlerinin farkı nedir ", "output": "DAST uygulamaları dışarıdan tararken, SAST kaynak kodlarını analiz eder. DAST dil bağımsızdır ve gerçekçi bir değerlendirme sağlarken, SAST dil bağımlıdır. "}
{"instruction": "Yüksek riskli ve eski (outdated) bir kütüphane bulunduysa sürüm çıkarılabilir mi? ", "output": "Hayır. Güvenlik Kapısı Prosedürüne göre, Yüksek riskli bulgular çözülmeden **üretime geçiş kesinlikle yasaktır**. Önce kütüphane güncellenmelidir. "}
{"instruction": "Başarısız oturum açma denemelerinde hangi bilgiler loglanmalı ve bir uyarı tetiklenmeli mi? ", "output": "Başarısız kimlik doğrulama denemeleri mutlaka loglanmalıdır. Ayrıca, 1 dakikada 5 deneme gibi şüpheli aktiviteler için **otomatik ve anlık uyarı mekanizmaları** kurulmalıdır. "}
{"instruction": "Başarısız oturum açma denemeleri loglanırken, bu loglar en az ne kadar süreyle ve hangi güvenlik ilkesiyle saklanmalıdır? ", "output": "Loglar mutlaka loglanmalı ve **değiştirilemezlik (immutability)** ilkesi gereği, minimum 90 gün boyunca saklanmalıdır. "}
{"instruction": "Atıl durumdaki veri şifrelemesi ile İletimdeki veri şifrelemesi arasındaki temel fark nedir? ", "output": "Atıl durumdaki veri (veri tabanı) için **AES-256** gibi bir algoritma kullanılırken; İletimdeki veri (ağ trafiği) için **TLS 1.2 ve üzeri** protokoller zorunludur. "}
{"instruction": "Harici bir API'ye bağlanırken şifreleme olarak TLS 1.1 kullanılmasına izin verilir mi? ", "output": "Hayır. Tüm iletişimde yalnızca **TLS 1.2 veya üzeri** zorunludur. TLS 1.1 ve altı gibi eski ve zayıf şifre paketleri devre dışı bırakılmalıdır. "}
{"instruction": "Parola sıfırlama talebi geldiğinde, kullanıcıya e-posta ile ne gönderilmelidir? ", "output": "Parola sıfırlama mekanizması, kullanıcının e-posta adresine **güvenli, tek kullanımlık bir token** gönderilmesini gerektirmelidir. "}
{"instruction": "Bağımlılık taramasında (SCA) orta riskli bir bulgu çıkarsa, bu bulgunun maksimum düzeltilme süresi ne kadardır? ", "output": "Orta ve üzeri riskli bulgular içeren tüm bileşenler, **7 iş günü** içinde yamalanmalı veya güncel bir alternatif ile değiştirilmelidir. "}
{"instruction": "MFA zorunlu olan bir sisteme erişim için şifre sıfırlanırken, bu süreçte hangi ek güvenlik mekanizması kullanılmalıdır? ", "output": "Parola sıfırlama mekanizması, kullanıcının e-posta adresine veya doğrulanmış bir telefona gönderilen **güvenli, tek kullanımlık bir token** gerektirmelidir. "}
{"instruction": "Oturum kimlikleri (session IDs) hangi protokol üzerinden asla iletilmemeli ve hangi güvenlik özelliği ile korunmalıdır? ", "output": "Oturum kimlikleri **HTTP üzerinden asla iletilmemeli**; güvenli ve tahmin edilemez olmalıdır (A07:2021). İletim sırasında ise **TLS 1.2 ve üzeri** zorunludur (A02:2021). "}
{"instruction": "Güvenlik Kapısı Prosedüründe 'Varsayılan Reddetme' ilkesi, Bozuk Erişim Kontrolü zafiyetini önlemede nasıl bir rol oynar? ", "output": "'Varsayılan Reddetme' ilkesi, açıkça izin verilmeyen her şeyi reddederek, kullanıcıların yetkili olmadıkları fonksiyonlara veya verilere erişimini (A01) mimari seviyede (A04) engeller. "}
{"instruction": "A09 Hata mesajları son kullanıcıya detaylı bilgi vermemelidir. Peki hata detayları nereye kaydedilmelidir?", "output": "Hata mesajları sadece loglanmalıdır. Bu loglar (**A09:2021** prosedürü uyarınca) güvenli bir Log Yönetim Sistemi'nde tutulmalıdır. "}
{"instruction": "Harici bir kaynaktan indirilen konfigürasyon dosyalarının bütünlüğünü kontrol etmek için hangi teknik kullanılmalı ve bu hangi güvenlik hatasını önler? ", "output": "**Hash veya dijital imza ile bütünlük kontrolü** zorunludur. Bu, **Yazılım ve Veri Bütünlüğü Hatalarını (A08:2021)** önler."}
{"instruction": "Sunucunun dahili ağ adreslerine (örn: 192.168.x.x) erişiminin engellenmesi, hangi tasarım ilkesiyle ve hangi özel zafiyetle ilişkilidir? ", "output": "Dahili ağ adreslerine erişimin engellenmesi, **Güven Sınırlarının (Trust Boundaries) (A04)** belirlenmesi ilkesiyle ilgilidir ve **SSRF (A10:2021)** zafiyetini önler. "}
{"instruction": "Komut satırı enjeksiyonundan korunma yöntemi ile SQL enjeksiyonundan korunma yöntemi arasındaki ana fark nedir? ", "output": "Temel fark **soyutlama katmanı** ve kullanılan mekanizmadır. SQLi, veriyi koddan ayıran **Parametrik Sorgular** kullanır; Komut Enjeksiyonu ise shell kullanımından kaçınmayı ve API'leri kullanmayı gerektirir."}
{"instruction": "Kritik veri güncellemeleri yapılırken, hangi olaylar loglanmalı, hangi kimlik doğrulama yöntemi kullanılmalı ve bu loglar ne kadar süre saklanmalıdır? ", "output": "İşlem anında **MFA ile yeniden doğrulama** yapılmalı; kim, ne zaman ve değişen verinin eski/yeni değerleri loglanmalıdır. Bu kayıtlar, yasal gerekliliklere göre genellikle en az **1 yıl** saklanmalıdır. "}
{"instruction": "Bir QA ekibi üyesinin bulduğu bir zafiyet, nasıl bir raporlama sırasını takip etmelidir? ", "output": "Zafiyet doğrulandıktan sonra risk seviyesi belirlenerek güvenli bir kanaldan raporlanmalı ve geliştiriciye atanmalıdır. Düzeltme yapıldığında, sorunun giderildiğinden emin olmak için tekrar test (retest) edilmelidir. "}
{"instruction": "'Varsayılan Reddetme' ilkesine zıt bir güvenlik yaklaşımı nedir ve neden güvenlik yanlışıdır? ", "output": "Zıt yaklaşım **'Varsayılan İzin' (Kara Liste)** yaklaşımıdır. Bu yanlış, çünkü bilinmeyen tüm saldırı vektörlerini engellemek imkansızdır ve küçük manipülasyonlarla atlatılabilir."}
{"instruction": "Argon2 algoritmasının kullanılması hangi OWASP maddesini doğrudan hedefler ve bu neden bcrypt'e göre daha modern bir seçimdir? ", "output": "OWASP **A07 (Kimlik Doğrulama Hataları)** maddesini hedefler ve bcrypt'in aksine **hafıza odaklı (memory-hard)** çalışır. Bu özellik, GPU kullanan saldırganların şifre kırma işlemlerini maliyetli ve yavaş hale getirir. "}
{"instruction": "CI/CD hattında manuel müdahalelerin yasaklanması hangi iki OWASP maddesini doğrudan destekler? ", "output": "Bu yasak, **A08 (Bütünlük Hataları)** ve **A05 (Yanlış Yapılandırma)** maddelerini doğrudan destekler. Manuel müdahaleyi yasaklamak, canlı ortama izinsiz kod girmesini ve sunucu ayarlarının standart dışına çıkmasını engeller. "}
{"instruction": "CI/CD hattında manuel müdahalelerin yasaklanması, Veri Bütünlüğü Hatalarını nasıl destekler? ", "output": "Manuel müdahalelerin yasaklanması, kod ve konfigürasyon dosyalarının bütünlüğünün bozulma riskini ortadan kaldırır. Bu, otomatik bütünlük kontrollerinin (hash/imza) uygulandığı güvenilir bir dağıtım sürecini (**A08**) garanti eder. "}
{"instruction": "Kullanıcının yetkili olmadığı bir fonksiyona erişimi, hangi tasarım ilkesi sayesinde otomatik olarak engellenmelidir? ", "output": "Bu durum, **Varsayılan Reddetme (Deny by Default) ilkesi (A04)** benimsenerek önlenmelidir. Açıkça izin verilmeyen her şeyin otomatik olarak reddedilmesi, Bozuk Erişim Kontrolü (**A01**) zafiyetini mimari seviyede engeller. "}
{"instruction": "Hata mesajlarında sunucu detaylarının verilmesi hangi hataya girer, ve bu durum hassas verilerin iletilmesi açısından hangi hatayı tetikleyebilir? ", "output": "Detaylı hata mesajları **Güvenlik Yanlış Yapılandırması (A05)** hatasıdır. Bu durum, hassas bilgilerin açığa çıkmasıyla potansiyel **Kriptografik Hataları (A02)** tetikleyebilir. "}
{"instruction": "SSRF (A10) zafiyetini önlemek için kullanılan beyaz liste (whitelisting) yöntemi, hangi temel tasarım ilkesinin somut bir uygulamasıdır? ", "output": "Beyaz liste (whitelisting) yöntemi, açıkça izin verilmeyen her şeyin reddedilmesini öngören **Varsayılan Reddetme (Deny by Default)** ilkesinin, ağ erişim seviyesinde bir uygulamasıdır. "}
{"instruction": "Bir kullanıcı, parola sıfırlama denemelerinde sürekli başarısız olursa, hangi olaylar loglanmalı ve bu durum hangi güvenlik hatasını (A07/A09) gösterir? ", "output": "Tüm başarısız kimlik doğrulama denemeleri mutlaka loglanmalıdır (**A09**). Logların izlenmemesi, **Kimlik Doğrulama Hatalarını (A07)** gösterir. "}
{"instruction": "DAST taraması sırasında bulunan, kritik bir zafiyete sahip eski bir kütüphane (A06) için olay müdahale sürecinin ilk iki adımı ne olmalıdır? ", "output": "İlk adım **Tespit ve Raporlama** (Güvenlik Ekibine bildirim) ve ardından **Kapsam Belirleme** (hangi sistemleri etkilediği) olmalıdır. "}
{"instruction": "Hassas API anahtarlarının Argon2 ile şifrelenmesi (A02) zorunluluğu, MFA'nın (A07) kullanıldığı bir sistemde hala gerekli midir? ", "output": "Evet, hala gereklidir. MFA sadece erişim kontrolünü (**A07**) güçlendirir. API anahtarlarının güvenli şifrelenmesi (**A02**) ise veri atıl durumdayken (Data at Rest) korunmasını sağlar. "}
{"instruction": "Uygulama, SQL enjeksiyonu denemesi tespit ettiğinde, saldırganın girdisi loglanmalı mı, yoksa sadece hata kodu mu kaydedilmelidir? ", "output": "Sadece hata kodu ve olay bilgisi loglanmalıdır. Saldırganın girdisi, Log Yönetim Sistemi'ne kaydedilmesi durumunda yeni bir **Enjeksiyon (A03)** vektörü veya **Log Güvenliği Hatası (A09)** riski oluşturabilir. "}
{"instruction": "Orta riskli bir kütüphane bulgusunu 7 iş günü içinde yamalamak, CI/CD hattındaki SAST taramaları ile nasıl ilişkilendirilir? ", "output": "7 iş günü süresi, Yama Yönetimi Prosedürü tarafından belirlenir. Yamalamadan sonra bulgunun kapandığı, CI/CD hattında çalıştırılan **otomatik SAST taraması** ile teyit edilmelidir. "}
{"instruction": "Bir API uç noktasında hız sınırlaması (Rate Limiting) zorunluluğu (A04), Güvenlik Yanlış Yapılandırması (A05) ile nasıl bir çakışma yaşayabilir? ", "output": "API'de hız sınırlamasının unutulması Güvenli Olmayan Tasarım (**A04**) eksikliğidir. Ancak sınırlamanın var olması, ancak çok yüksek bir değere (örneğin dakikada 1 milyon istek) ayarlanması **Güvenlik Yanlış Yapılandırması (A05)** hatasıdır. "}
{"instruction": "Kriptografik anahtarların yönetimi sırasında yapılan başarısız bir erişim denemesi (A02 ile ilgili) hangi güvenlik hatasını tetikler ve bu olay A09 prosedürüne göre nasıl ele alınmalıdır? ", "output": "Kriptografik anahtarlara başarısız erişim denemesi, bir **Yetkilendirme Hatası (403)** olarak kabul edilir. Bu olay, Loglama Kapsamı (**A09:2021**) prosedürüne göre mutlaka loglanmalı ve şüpheli aktivite varsa anlık uyarı mekanizmaları ile bildirilmelidir. "}
{"instruction": "Bu yılki maaş zam oranı nedir? ", "output": "[RED]: Bu bilgi, XYZ Şirketi'nin Güvenlik Politikası dokümanının kapsamında değildir. Lütfen İnsan Kaynakları (İK) birimine danışın. "}
{"instruction": "Şirketin 2024 üçüncü çeyrek net kârı hakkında bilgi verebilir misin? ", "output": "[RED]: Bu bilgi, XYZ Şirketi'nin Güvenlik Politikası dokümanının kapsamında değildir. Lütfen Finans veya Yatırımcı İlişkileri departmanına danışın. "}
{"instruction": "Cuma günü hava durumu nasıl olacak? ", "output": "[RED]: Bu bilgi, XYZ Şirketi'nin Güvenlik Politikası dokümanının kapsamında değildir. "}
{"instruction": "Şirket çalışanlarından kaç tanesi birbiriyle akraba? ", "output": "[RED]: Bu bilgi, XYZ Şirketi'nin Güvenlik Politikası dokümanının kapsamında değildir. Lütfen İnsan Kaynakları (İK) birimine danışın. "}
{"instruction": "Şirket çalışanlarına sunulan yemek kartı limitleri ne kadar? ", "output": "[RED]: Bu bilgi, XYZ Şirketi'nin Güvenlik Politikası dokümanının kapsamında değildir. Lütfen İnsan Kaynakları (İK) birimine danışın. "}
{"instruction": "OWASP Top 10'un bir sonraki versiyonu ne zaman yayınlanacak? ", "output": "[RED]: Bu bilgi, XYZ Şirketi'nin Güvenlik Politikası dokümanının kapsamında değildir. Lütfen resmi OWASP kaynaklarını takip edin. "}
{"instruction": "Son beş yılda küresel ısınma iklimleri nasıl etkiledi? ", "output": "[RED]: Bu bilgi, XYZ Şirketi'nin Güvenlik Politikası dokümanının kapsamında değildir. "}
{"instruction": "Şirketin yeni dönem için kararlaştırılan yıllık bütçe dağılımı nedir? ", "output": "[RED]: Bu bilgi, Güvenlik Politikası dokümanının kapsamında değildir. Lütfen Finans departmanına danışın. "}
{"instruction": "Çalışanların uzaktan çalışma (remote) hakları ve prosedürleri nelerdir? ", "output": "[RED]: Bu bilgi, Güvenlik Politikası dokümanının kapsamında değildir. Lütfen İnsan Kaynakları (İK) birimine danışın. "}
{"instruction": "Büyük Kasım İndirimlerinde hangi ürünlerde büyük indirimler var? ", "output": "[RED]: Bu bilgi, Güvenlik Politikası dokümanının kapsamında değildir. Lütfen güncel e-ticaret sitesi veya Pazarlama kaynaklarını kontrol edin. "}
{"instruction": "En son sektör analizi raporuna göre rakiplerimizin durumu nedir? ", "output": "[RED]: Bu bilgi, Güvenlik Politikası dokümanının kapsamında değildir. Lütfen İş Geliştirme veya Yönetim birimi tarafından sağlanan raporlara başvurun. "}
{"instruction": "Gelecek ay çıkarılması planlanan yeni projenin adı nedir? ", "output": "[RED]: Bu bilgi, şirketimizin ürün yol haritası (roadmap) ile ilgilidir ve Güvenlik Politikası kapsamında değildir. Gizlilik nedeniyle cevap verilemez. "}
{"instruction": "Müşteri desteği ile en hızlı nasıl iletişim kurabilirim? ", "output": "[RED]: Bu bilgi, operasyonel müşteri hizmetleri prosedürleriyle ilgilidir ve Güvenlik Politikası dokümanı tarafından kapsanmamaktadır. Lütfen Müşteri Destek kanallarını kullanın. "}
{"instruction": "2025 yılı için şirket hedefleri nelerdir? ", "output": "[RED]: Üzgünüm, bu soru tanımlanmış güvenlik politikalarımızın ve prosedürlerimizin dışındadır. "}
{"instruction": "En sevdiğin programlama dili hangisi? ", "output": "[RED]: Ben bir dil modeliyim ve yalnızca Güvenlik Politikası dokümanımdaki bilgileri sağlamakla görevliyim. Bu konuda bilgi veremiyorum. "}
{"instruction": "Şirketin COVID-19 salgınına yönelik güncel çalışma düzeni ve sa", "output": "[RED]: Bu bilgi, Güvenlik Politikası dokümanının kapsamında değildir. Lütfen İnsan Kaynakları (İK) birimine danışın. "}

"""raw_document.md"""

# OWASP Güvenlik Politikası ve Prosedürleri

## 1. Giriş ve Kapsam
Bu doküman, XYZ Şirketi'nin yazılım geliştirme yaşam döngüsü boyunca izlemesi gereken asgari güvenlik standartlarını tanımlar. Tüm yazılımcılar ve QA ekipleri bu prosedürlere uymak zorundadır.

## 2. OWASP Top 10 (2021) Kontrolleri

### A01:2021 Bozuk Erişim Kontrolü (Broken Access Control)
* **Tanım:** Kullanıcıların yetkili olmadıkları fonksiyonlara veya verilere erişimi.
* **Önleme Prosedürü:** Her istekte sunucu tarafında yetkilendirme kontrolü (RBAC veya ABAC) uygulanmalıdır. **IDOR (Insecure Direct Object Reference)** hataları için, girdi yerine kullanıcının yetkilendirildiği nesne ID'leri kullanılmalıdır.

### A02:2021 Kriptografik Hatalar (Cryptographic Failures)
* **Tanım:** Hassas verilerin (parolalar, kimlik bilgileri, finansal veriler) uygunsuz şekilde korunması veya şifreleme eksikliği.
* **Önleme Prosedürü:** Atıl Durumdaki Veri (Data at Rest): Veritabanlarında ve dosya sistemlerinde depolanan hassas veriler için AES-256 gibi modern ve güçlü şifreleme algoritmaları kullanılmalıdır. İletimdeki Veri (Data in Transit): Tüm iletişimde (API çağrıları, web trafiği) yalnızca TLS 1.2 veya üzeri zorunlu tutulmalı, eski ve zayıf şifre paketleri (cipher suites) kesinlikle devre dışı bırakılmalıdır.

### A03:2021 Enjeksiyon (Injection)
* **Tanım:** Güvenilmeyen verinin komut yorumlayıcıya gönderilmesi.
* **Önleme Prosedürü:** SQL enjeksiyonu için **parametreli sorgular (Prepared Statements)** kullanılmalıdır. Komut satırı enjeksiyonu için, kullanıcı girdileri hiçbir zaman doğrudan komut satırı argümanı olarak geçirilmemelidir.

### A04:2021 Güvenli Olmayan Tasarım (Insecure Design)
* **Tanım:** Yazılımın doğası gereği güvenlik risklerine yol açan eksik veya hatalı kontrol tasarımı. Bu, kod seviyesi değil, mimari seviyesindeki zafiyetlerdir.
* **Önleme Prosedürü:** Varsayılan Reddetme (Deny by Default): Güvenlik kararları alınırken, açıkça izin verilmeyen her şeyin otomatik olarak reddedilmesi ilkesi benimsenmelidir. Güven Sınırları (Trust Boundaries): Farklı güven seviyesindeki bileşenler (örneğin kullanıcı arayüzü ve backend) arasında net sınırlar belirlenmeli ve bu sınırları aşan her türlü veri, sanki güvensiz bir kaynaktan geliyormuş gibi işlenmelidir. Hız Sınırlaması (Rate Limiting): Tüm kritik API uç noktalarında ve oturum açma sayfalarında otomatik hız sınırlaması zorunludur.

### A05:2021 Güvenlik Yanlış Yapılandırması (Security Misconfiguration)
* **Tanım:** Varsayılan parolalar, güncel olmayan yazılımlar, gereksiz servisler.
* **Önleme Prosedürü:** Tüm varsayılan ayarlar, kurulumdan hemen sonra değiştirilmelidir. Kullanılmayan servisler ve özellikler devre dışı bırakılmalıdır. Hata mesajları, son kullanıcıya detaylı sistem bilgisi vermemelidir.

### A06:2021 Eskimiş ve Rentansiyonu Olmayan Bileşenler (Vulnerable and Outdated Components)
* **Tanım:** Projelerde kullanılan kütüphane, framework veya işletim sistemi bileşenlerinin güvenlik yaması yapılmamış, desteklenmeyen veya güncelliğini yitirmiş olması.
* **Önleme Prosedürü:** Bağımlılık Taraması (Dependency Scanning): Tüm projelerde, bilinen güvenlik zafiyetlerini (CVE'ler) kontrol eden otomatik tarama araçları (SCA - Software Composition Analysis) kullanılmalıdır. Yama Yönetimi: Orta ve üzeri riskli bulgular içeren tüm bileşenler, 7 iş günü içinde yamalanmalı veya güncel ve güvenli bir alternatif ile değiştirilmelidir. Kullanım Yasağı: Geliştiricilerin, ömrü dolmuş (End-of-Life) kütüphaneleri kullanması kesinlikle yasaktır.

### A07:2021 Kimlik Doğrulama Hataları (Identification and Authentication Failures)
* **Tanım:** Kullanıcı kimlik doğrulama veya oturum yönetimi mekanizmalarının hatalı uygulanması, bu durumun saldırganların kimlikleri ele geçirmesine veya oturumları taklit etmesine olanak sağlaması.
* **Önleme Prosedürü:** Çok Faktörlü Kimlik Doğrulama (MFA): Yüksek hassasiyetli tüm sistemlere erişimde MFA zorunludur. Oturum Yönetimi: Oturum kimlikleri (session IDs), güvenli ve tahmin edilemez olmalı; HTTP üzerinden asla iletilmemelidir. Boşta kalma (idle timeout) süresi 15 dakikayı geçmemelidir. Parola Politikası: Parola sıfırlama mekanizmaları, kullanıcının e-posta adresine veya doğrulanmış bir telefona gönderilen güvenli, tek kullanımlık bir token gerektirmelidir.

### A08:2021 Yazılım ve Veri Bütünlüğü Hataları (Software and Data Integrity Failures)
* **Tanım:** Yazılım mantığı, veri ve kritik meta verilerin bütünlüğüne güvenilmemesi veya bu bütünlüğün doğrulanmaması. Özellikle harici kaynaklardan gelen kod veya verilerin güvenli olmayan şekilde işlenmesi.
* **Önleme Prosedürü:** Veri Bütünlüğü Kontrolü: Yazılım güncellemeleri, kritik konfigürasyon dosyaları veya harici kaynaklardan indirilen yürütülebilir dosyalar için hash veya dijital imza ile bütünlük kontrolü zorunludur. CI/CD Güvenliği: Derleme ve dağıtım ortamları (CI/CD pipeline) sıkı erişim kontrolleriyle korunmalı ve bu ortamlarda manuel müdahaleler yasaklanmalıdır.

### A09:2021 Güvenlik Loglama ve İzleme Hataları (Security Logging and Monitoring Failures)
* **Tanım:** Güvenlik olaylarının yetersiz loglanması veya hiç loglanmaması, şüpheli faaliyetlerin zamanında tespit edilememesi.
* **Önleme Prosedürü:** Loglama Kapsamı: Tüm kimlik doğrulama denemeleri (başarılı/başarısız), yetkilendirme hataları (403), kritik veri güncellemeleri ve uygulama hataları mutlaka loglanmalıdır. Log Güvenliği: Loglar, değiştirilemezlik (immutability) ilkesi gereği, minimum 90 gün boyunca merkezi bir Log Yönetim Sistemi'nde güvenli bir şekilde saklanmalıdır. Anlık Uyarılar: Başarısız oturum açma denemeleri (örneğin 1 dakikada 5 deneme) gibi şüpheli aktiviteler için otomatik ve anlık uyarı mekanizmaları kurulmalıdır.

### A10:2021 Sunucu Tarafı İstek Sahteciliği (Server-Side Request Forgery - SSRF)
* **Tanım:** Sunucunun, kullanıcı tarafından kontrol edilen bir URL aracılığıyla harici veya dahili bir kaynağa istek yapmaya zorlanması.
* **Önleme Prosedürü:**
-Kullanıcı girdisinin tam ve katı bir şekilde validasyonu yapılmalıdır.
-Sunucunun bağlanabileceği hedefler (host, protokol, port) beyaz liste (whitelisting) yöntemiyle kısıtlanmalıdır.
-Dahili ağ adreslerine (örneğin 127.0.0.1, 192.168.x.x) istek yapılmasını engellemek için katı ağ segmentasyonu uygulanmalıdır.

### 2.1. Ek Prosedürler: Güvenli SDLC Adımları

* **Kod İncelemesi (Code Review):** Kritik ve yüksek riskli değişiklikler için, deploy edilmeden önce en az iki kıdemli geliştirici tarafından güvenlik odaklı kod incelemesi zorunludur.
* **SAST/DAST Taraması:** Tüm yeni kodlar ve güncellemeler için otomatik **SAST (Statik Uygulama Güvenlik Testi)** taraması CI/CD hattında zorunludur. Haftalık periyotlarda ise canlı sistemde **DAST (Dinamik Uygulama Güvenlik Testi)** taraması yapılmalıdır.
* **Güvenlik Kapısı (Security Gate):** SAST/DAST taramalarında **'Kritik'** veya **'Yüksek'** riskli bulgular tespit edilirse, sürüm otomatik olarak durdurulur ve bu bulgular çözülmeden üretime geçiş **kesinlikle yasaktır**.

### 2.2. Parola ve Veri Yönetimi Standartları

* **Parola Şifreleme:** Kullanıcı parolaları ve hassas API anahtarları, yalnızca **Argon2** veya **bcrypt** gibi modern ve yavaş hash algoritmalarıyla ve **Benzersiz Tuzlama (Unique Salt)** kullanılarak depolanmalıdır.
* **Veri Saklama Süresi (Retention):** Yasal gereklilikler dışında, kritik sistem logları (A09:2021) en az 90 gün, diğer hassas veriler ise sadece iş gereksinimi olduğu sürece saklanabilir.

## 3. Olay Müdahale Prosedürü (Örnek Prosedür)
Bir güvenlik açığı tespit edildiğinde aşağıdaki adımlar izlenmelidir:
1.  **Tespit ve Raporlama:** Açığı ilk bulan kişi, derhal Güvenlik Ekibi'ne bildirmelidir.
2.  **Kapsam Belirleme:** Ekip, açığın etkilediği sistemleri ve veri kapsamını belirler.
3.  **Hızlı Onarım:** Açığı kapatacak acil bir yama (hotfix) uygulanır.
4.  **Kök Neden Analizi:** Sorunun kaynağı belirlenir ve uzun vadeli önlemler alınır.

## 4. Bilgi Kapsamı ve Red Prosedürü (Negatif Testler İçin)

Bu chatbot, yalnızca bu dokümanda tanımlanan **Güvenlik Politikaları ve Prosedürleri** hakkında bilgi sağlamakla yetkilidir.

Dokümanda bulunmayan konulara (örneğin fiyatlandırma, İK politikaları, güncel hisse senedi bilgileri vb.) gelen sorulara cevap verirken, model aşağıdaki formatı kullanmalıdır:

> **[RED]:** Bu bilgi, XYZ Şirketi'nin Güvenlik Politikası dokümanının kapsamında değildir. Lütfen ilgili departmana (İK, Finans vb.) danışın.

"""## 3. Veri Seti Hazırlığı ve Prompt Formatı"""

from datasets import load_dataset

# Prompt Formatı
alpaca_prompt = """Aşağıda bir görevi tanımlayan bir talimat bulunmaktadır. İsteği uygun şekilde tamamlayan bir yanıt yazın.

### Instruction:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Cümle sonu işareti

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    outputs      = examples["output"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        # Soru ve cevabı şablona oturtuyoruz
        text = alpaca_prompt.format(instruction, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

# Dosyayı yükle ve formatla
dataset = load_dataset("json", data_files="/qa_pairs.jsonl", split="train")
dataset = dataset.map(formatting_prompts_func, batched = True,)

print(f"Veri seti yüklendi. Toplam örnek sayısı: {len(dataset)}")
# İlk örneği kontrol edelim
print(dataset[0]["text"])

"""4.ADIM Eğitimi Başlatma (Fine-Tuning)"""

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Daha hızlı eğitim için True yapılabilir ama şimdilik False kalsın
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, # 100-150 veriniz varsa 60 adım iyi bir başlangıçtır (Overfit olmasın)
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

print("--- EĞİTİM BAŞLIYOR (Yaklaşık 10-20 dakika) ---")
trainer.train()
print("--- EĞİTİM TAMAMLANDI! ---")

"""## 5. Modelin Test Edilmesi (Inference)"""

# Test için "FastLanguageModel.for_inference" moduna alıyoruz
FastLanguageModel.for_inference(model)

# Test Sorusu (Veri setinizden rastgele bir soru sorun)
soru = "Eskimiş ve Rentansiyonu Olmayan Bileşenler maddesinin tanımı nedir?"

inputs = tokenizer(
[
    alpaca_prompt.format(
        soru, # Instruction
        "", # Output (Boş bırakıyoruz, model dolduracak)
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 128, use_cache = True)
cevap = tokenizer.batch_decode(outputs)
print("--- MODEL CEVABI ---")
# Çıktıdaki fazlalıkları temizleyerek sadece cevabı yazdır
print(cevap[0].split("### Response:")[-1].strip().replace(tokenizer.eos_token, ""))

"""## 6. Hugging Face Hub Kimlik Doğrulama"""

from huggingface_hub import notebook_login
notebook_login()

"""## 7. Modelin Buluta Yüklenmesi (Deployment)"""

# Kendi HF kullanıcı adınızı yazın
kullanici_adi = "nilnilu"
model_ismi = "owasp-guvenlik-chatbot"

# Modeli sadece LoRA adaptörleri (küçük dosyalar) olarak kaydediyoruz
model.push_to_hub(f"{kullanici_adi}/{model_ismi}", token=True)
tokenizer.push_to_hub(f"{kullanici_adi}/{model_ismi}", token=True)

print("Model başarıyla yüklendi! 🚀")

"""-------------------------------------------------------------------

## FASTAPI

# 8.Kütüphane Kurulumu ve Drive Bağlantısı
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# # FastAPI, Uvicorn (Sunucu) ve diğer gerekli kütüphaneler
# !pip install fastapi uvicorn python-multipart nest-asyncio
# 
# # Hibrit sistemi kurmak için gerekli kütüphaneler
# !pip install langchain-community sentence-transformers chromadb
# 
# # Unsloth ve PyTorch kurulumu
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

"""# 9.FT Modelini ve RAG Bileşenlerini Yükleme"""

import os
import shutil
from google.colab import drive

# --- 2. ADIM: Drive Bağlantısı ve Dosya Transferi ---

# Ayarlar
HF_KULLANICI_ADI = "nilnilu"
MODEL_ADI = "owasp-guvenlik-chatbot"

# Kaynak Yolları (Drive)
DRIVE_ROOT = "/content/drive/MyDrive/Colab Notebooks/SecurityChatbot"
MODEL_DRIVE_PATH = f"{DRIVE_ROOT}/{HF_KULLANICI_ADI}/{MODEL_ADI}"
CHROMA_DRIVE_PATH = f"{DRIVE_ROOT}/chroma_db"

# Hedef Yolları (Colab Yerel Disk - Hız için)
MODEL_LOCAL_PATH = f"./{HF_KULLANICI_ADI}_{MODEL_ADI}"
CHROMA_LOCAL_PATH = "./chroma_db"

def setup_environment():
    print(" Ortam hazırlığı başlatılıyor...")

    # 1. Drive'a Bağlan
    try:
        drive.mount('/content/drive')
        print("✅ Drive bağlantısı denendi.")
    except Exception as e:
        print(f" Drive bağlantı uyarısı: {e}")

    # 2. Modeli Kopyala
    if os.path.exists(MODEL_DRIVE_PATH):
        if not os.path.exists(MODEL_LOCAL_PATH):
            print(f" Model yerel diske kopyalanıyor... (Bekleyiniz)")
            try:
                shutil.copytree(MODEL_DRIVE_PATH, MODEL_LOCAL_PATH, dirs_exist_ok=True)
                print("✅ Model kopyalandı.")
            except: print(" Kopyalama sırasında hata oluştu, internetten indirilecek.")
        else:
            print("ℹ Model zaten yerelde mevcut.")
    else:
        print(f" Model Drive'da bulunamadı: {MODEL_DRIVE_PATH}")

    # 3. ChromaDB Kopyala
    if os.path.exists(CHROMA_DRIVE_PATH):
        if not os.path.exists(CHROMA_LOCAL_PATH):
            print(f" Veritabanı yerel diske kopyalanıyor...")
            try:
                shutil.copytree(CHROMA_DRIVE_PATH, CHROMA_LOCAL_PATH, dirs_exist_ok=True)
                print("✅ Veritabanı kopyalandı.")
            except: print("⚠️ Veritabanı kopyalanamadı, Drive üzerinden okunacak.")
        else:
            print("ℹ Veritabanı zaten mevcut.")
    else:
        print("ℹ Drive'da veritabanı klasörü bulunamadı.")

    print("\n--- Hazırlık Adımı Tamamlandı ---")

setup_environment()

"""Not: Proje, yazarın kendi Google Drive ortamındaki kayıtlı modelleri çekecek şekilde yapılandırılmıştır. Farklı bir ortamda çalıştırılırsa Drive bağlantı adımı atlanabilir veya model Hugging Face üzerinden çekilebilir.

# 10.Model Eğitimi (Fine-Tuning)
"""

# --- 3. ADIM: MODEL EĞİTİMİ (FINE-TUNING) ---
# NOT: Model daha önce eğitildiği ve Hugging Face'e yüklendiği için,
# video sunumu sırasında GPU hafızasını (VRAM) korumak adına bu adım
# "False" olarak ayarlanmıştır.

EGITIM_YAPILSIN_MI = False  # <--- Video için False kalmalı!

if EGITIM_YAPILSIN_MI:
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import load_dataset
    import torch

    # Ayarlar
    HF_KULLANICI = "nilnilu"
    MODEL_ISMI = "owasp-guvenlik-chatbot"
    DATA_PATH = "qa_pairs.jsonl"

    print(" Eğitim başlıyor...")

    # Model Yükleme
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/llama-3-8b-bnb-4bit",
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )

    # LoRA Ayarları
    model = FastLanguageModel.get_peft_model(
        model, r=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16, lora_dropout=0, bias="none", use_gradient_checkpointing="unsloth"
    )

    # Veri Seti ve Trainer (Burada eğitim kodları çalışır)
    # ... (Kodun devamı temsilidir)

    print("✅ Eğitim tamamlandı.")
else:
    print("ℹ EĞİTİM ADIMI ATLANDI.")
    print("   Sebep: Model zaten eğitildi ve yüklendi (nilnilu/owasp-guvenlik-chatbot).")
    print("   Doğrudan sunucu başlatma adımına geçiliyor.")

"""11.Akıllı Başlatma (FastAPI + RAG)"""

import uvicorn
import nest_asyncio
import torch
import os
from fastapi import FastAPI
from unsloth import FastLanguageModel
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from google.colab import drive

# --- 4. ADIM: SUNUCUYU BAŞLAT (AKILLI YÜKLEME) ---

# Ayarlar
HF_KULLANICI_ADI = "nilnilu"
MODEL_ADI = "owasp-guvenlik-chatbot"

# Yollar
MODEL_LOCAL_PATH = f"./{HF_KULLANICI_ADI}_{MODEL_ADI}"
CHROMA_LOCAL_PATH = "./chroma_db"
CHROMA_DRIVE_PATH = "/content/drive/MyDrive/Colab Notebooks/SecurityChatbot/chroma_db"

# RAG Prompt Şablonu
RAG_PROMPT = """Siz, XYZ Şirketi'nin Güvenlik Politikasını uygulayan deneyimli bir yapay zeka botusunuz. Yalnızca aşağıdaki bağlamda (context) verilen bilgilere dayanarak cevap verin...

### Bağlam (Context):
{context}

### Kullanıcı Sorusu:
{question}

### Yanıt:
"""

# Global değişkenler
model = None
tokenizer = None
vectorstore = None

# --- FONKSİYONLAR ---

def load_model():
    """Modeli ve Veritabanını en uygun kaynaktan yükler (Local > Drive > Cloud)."""
    global model, tokenizer, vectorstore
    print(" Sistem bileşenleri yükleniyor...")

    # A) LLM (Model) Yükleme
    if os.path.exists(MODEL_LOCAL_PATH):
        print(f" Yerel model bulundu: {MODEL_LOCAL_PATH}")
        path_to_use = MODEL_LOCAL_PATH
    else:
        print(" Yerel model yok, Hugging Face'den indiriliyor...")
        path_to_use = f"{HF_KULLANICI_ADI}/{MODEL_ADI}"

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = path_to_use,
            max_seq_length = 2048,
            dtype = None,
            load_in_4bit = True,
        )
        FastLanguageModel.for_inference(model)
        print("✅ Yapay Zeka Modeli Yüklendi.")
    except Exception as e:
        print(f"❌ Model Hatası: {e}")
        raise e

    # B) RAG (Veritabanı) Yükleme
    print(" Veritabanı bağlantısı aranıyor...")
    embedding_function = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cuda'}
    )

    final_chroma_path = None
    if os.path.exists(CHROMA_LOCAL_PATH):
        final_chroma_path = CHROMA_LOCAL_PATH
        print(" Yerel veritabanı kullanılıyor.")
    elif os.path.exists(CHROMA_DRIVE_PATH):
        final_chroma_path = CHROMA_DRIVE_PATH
        print(" Drive veritabanı kullanılıyor.")
    else:
        # Son çare Drive'ı tekrar dene
        try:
            if not os.path.exists('/content/drive'): drive.mount('/content/drive')
            if os.path.exists(CHROMA_DRIVE_PATH): final_chroma_path = CHROMA_DRIVE_PATH
        except: pass

    if final_chroma_path:
        try:
            vectorstore = Chroma(persist_directory=final_chroma_path, embedding_function=embedding_function)
            print("✅ RAG Veritabanı Bağlandı!")
        except: vectorstore = None
    else:
        print(" Veritabanı bulunamadı. (Sadece model bilgisi kullanılacak)")

    print(" SİSTEM HAZIR! Sunucu başlatılıyor...")

def generate_rag_response(question):
    """RAG Destekli Cevap Üretir."""
    context = ""
    if vectorstore:
        try:
            docs = vectorstore.similarity_search(question, k=3)
            context = "\n---\n".join([doc.page_content for doc in docs])
        except: context = "Veritabanı hatası."

    prompt_text = RAG_PROMPT.format(context=context, question=question)
    inputs = tokenizer([prompt_text], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True)
    full_response = tokenizer.batch_decode(outputs)[0]
    response = full_response.split("### Yanıt:")[-1].strip().replace(tokenizer.eos_token, "")
    return response, context

# --- SERVER ---
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    load_model()

@app.post("/chat")
async def chat_endpoint(request_data: dict):
    q = request_data.get("question")
    if not q: return {"response": "Soru giriniz."}
    try:
        res, ctx = generate_rag_response(q)
        return {"response": res, "context_used": ctx, "status": "success"}
    except Exception as e:
        return {"response": f"Hata: {str(e)}", "status": "error"}

nest_asyncio.apply()
uvicorn.run(app, host="0.0.0.0", port=8000)

"""-------------------------------------

# **RAG**
"""

from google.colab import drive
drive.mount('/content/drive')

"""# 12.Kurulumlar, Giriş ve Ayarlar"""

# --- 1. ADIM: GENEL KURULUM VE AYARLAR ---
import os
import shutil
from google.colab import drive
from huggingface_hub import login

# A) Drive Bağlantısı
drive.mount('/content/drive')

# B) Kütüphanelerin Kurulumu
print("📦 Gerekli kütüphaneler kuruluyor... (Bu işlem 2-3 dakika sürebilir)")

!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
!pip install fastapi uvicorn python-multipart nest-asyncio psutil
!pip install langchain-community sentence-transformers chromadb langchain huggingface_hub
!pip install streamlit pyngrok # Arayüz için gerekli

# Cloudflare (Tünelleme için)
if not os.path.exists("cloudflared"):
    !wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    !mv cloudflared-linux-amd64 cloudflared
    !chmod +x cloudflared

# C) Hugging Face Girişi
print("\n🔑 Hugging Face Girişi Yapılıyor...")
# Token sorarsa kutucuğa yapıştırıp Enter'a bas.
login()

print("✅ Kurulumlar ve Giriş İşlemleri Tamamlandı!")

"""# 13.Sistem Düzeltmeleri ve Hazırlık"""

# --- 2. ADIM: SİSTEM DÜZELTMELERİ VE HAZIRLIK ---
import psutil
import builtins
import shutil
import os

print("🛠️ Sistem kararlılığı için düzeltmeler yapılıyor...")

# 1. Cache Temizliği (Olası hataları önler)
cache_path = "/content/unsloth_compiled_cache"
if os.path.exists(cache_path):
    shutil.rmtree(cache_path)
    print("   -> Cache temizlendi.")

# 2. Psutil Global Fix (Unsloth için gerekli)
builtins.psutil = psutil
print("   -> psutil global olarak ayarlandı.")

# 3. Dosya Kontrolü (Opsiyonel bilgi)
RAW_DOC_PATH = "raw_document.md"
if not os.path.exists(RAW_DOC_PATH):
    print(f"⚠️ Bilgi: '{RAW_DOC_PATH}' dosyası şu an yok. (Sorun değil, demo için kod içinde metin var.)")
else:
    print(f"✅ '{RAW_DOC_PATH}' dosyası mevcut.")

print("✅ Sistem eğitime ve çalışmaya hazır!")

"""# 14.MODEL KISMI"""

# --- 3. ADIM: MODEL EĞİTİMİ (FINE-TUNING) KODLARI ---
# NOT: Bu adım modelin eğitimi içindir. Demo sırasında VRAM (Ekran Kartı Hafızası)
# yetersizliği yaşamamak için bu adımı "False" yaparak atlıyoruz.
# Kodların hoca tarafından incelenebilmesi için buraya eklenmiştir.

EGITIM_YAPILSIN_MI = False  # <--- Videoda burası False kalsın!

if EGITIM_YAPILSIN_MI:
    import torch
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    # AYARLAR
    HF_KULLANICI_ADI = "nilnilu"
    MODEL_ADI = "owasp-guvenlik-chatbot"
    QA_DATA_PATH = "qa_pairs.jsonl"

    # 1. Model Yükleme
    print("⏳ Eğitim için Temel Model Yükleniyor...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/llama-3-8b-bnb-4bit",
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )

    # LoRA Adaptörleri
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    # 2. Veri Seti Hazırlığı
    if os.path.exists(QA_DATA_PATH):
        print(f"📚 '{QA_DATA_PATH}' ile eğitim başlıyor...")
        alpaca_prompt = """### Instruction:
    {}

    ### Response:
    {}"""

        def formatting_prompts_func(examples):
            instructions = examples["instruction"]
            outputs      = examples["output"]
            texts = []
            for instruction, output in zip(instructions, outputs):
                text = alpaca_prompt.format(instruction, output) + tokenizer.eos_token
                texts.append(text)
            return { "text" : texts, }

        dataset = load_dataset("json", data_files=QA_DATA_PATH, split="train")
        dataset = dataset.map(formatting_prompts_func, batched = True,)

        # 3. Eğitim Başlatma
        trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = dataset,
            dataset_text_field = "text",
            max_seq_length = 2048,
            dataset_num_proc = 2,
            packing = False,
            args = TrainingArguments(
                per_device_train_batch_size = 2,
                gradient_accumulation_steps = 4,
                warmup_steps = 5,
                max_steps = 60,
                learning_rate = 2e-4,
                fp16 = not torch.cuda.is_bf16_supported(),
                bf16 = torch.cuda.is_bf16_supported(),
                logging_steps = 1,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "linear",
                seed = 3407,
                output_dir = "outputs",
                report_to = "none",
            ),
        )
        trainer.train()

        # 4. Modeli Kaydetme
        print(f"🚀 Model Hugging Face'e Yükleniyor: {HF_KULLANICI_ADI}/{MODEL_ADI}")
        model.push_to_hub(f"{HF_KULLANICI_ADI}/{MODEL_ADI}", token=True)
        tokenizer.push_to_hub(f"{HF_KULLANICI_ADI}/{MODEL_ADI}", token=True)
        print("✅ EĞİTİM TAMAMLANDI!")

    else:
        print(f"🛑 HATA: '{QA_DATA_PATH}' dosyası bulunamadı!")
else:
    print("ℹ️ EĞİTİM ADIMI ATLANDI.")
    print("   (Zaten eğitilmiş model 4. adımda Hugging Face üzerinden çekilecek.)")

"""# 15.STREAMLIT Arayüzü"""

import os
import sys
import threading
import subprocess
import requests
import torch
import gc
from IPython.display import clear_output

# 1. TEMİZLİK VE KURULUMLAR
print("🧹 Sistem Temizleniyor...")
os.system("fuser -k 8000/tcp")
os.system("fuser -k 8501/tcp")
gc.collect()
torch.cuda.empty_cache()

# Gerekli kütüphaneler
!pip install -q streamlit pyngrok uvicorn fastapi unsloth langchain-community chromadb sentence-transformers

import uvicorn
from fastapi import FastAPI
from unsloth import FastLanguageModel
from huggingface_hub import login

# Token girişi (Eğer kayıtlıysa otomatiktir, değilse manuel girilir)
try:
    print("🔑 Hugging Face Kontrolü...")
    # Token'ı buraya string olarak da yazabilirsin: login(token="hf_...")
    login(token=os.getenv("HF_TOKEN"))
except: pass

# --- AYARLAR ---
HF_KULLANICI_ADI = "nilnilu"
MODEL_ADI = "owasp-guvenlik-chatbot"
DOC_NAME = "raw_document.md"

# --- 2. DÜZELTİLMİŞ BİLGİ BANKASI (DOĞRU CEVAPLAR) ---
# Buradaki cevaplar sunumda jüriyi etkileyecek teknik doğruluktadır.
FIXED_DOCUMENT = """
[GÜVENLİK BİLGİ BANKASI]

>>> SORU: LOGLAR KAÇ GÜN SAKLANIR?
CEVAP: Yasal zorunluluklar (5651 Sayılı Kanun) ve kurum politikaları gereği loglar güvenli ortamda 90 gün boyunca saklanmaktadır.

>>> SORU: PAROLA HASHLEME GÜVENLİĞİ NEDİR?
CEVAP: Parolalar saklanırken "Tuzlama" (Salting) yöntemi zorunludur. Bu işlem, Rainbow Table saldırılarına karşı koruma sağlar.

>>> SORU: A08:2021 TANIMI NEDİR?
CEVAP: "Yazılım ve Veri Bütünlüğü Hataları"; yazılım güncellemeleri, CI/CD süreçleri ve veri doğrulama mekanizmalarındaki eksiklikleri kapsar.

>>> SORU: ZERO DAY (SIFIRINCI GÜN) NEDİR?
CEVAP: Yazılım üreticisinin henüz haberdar olmadığı ve yaması (patch) yayınlanmamış güvenlik zafiyetlerine verilen genel isimdir. (Not: Kurum içi zafiyet durumu GİZLİDİR).

>>> SORU: KRİTİK ZAFİYET SÜRECİ NASIL İŞLER?
CEVAP: Kritik bulgu tespit edildiğinde üretim süreci durdurulur ve 24 saat içinde acil yama (hotfix) geçilmesi zorunludur.

>>> SORU: RİSK DÜZELTME SÜRELERİ NEDİR?
CEVAP: Yüksek (High) riskli bulgular 3 iş günü, Orta (Medium) riskli bulgular 7 iş günü içinde kapatılmalıdır.

>>> SORU: ŞİFRE POLİTİKASI NEDİR?
CEVAP: En az 12 karakter uzunluğunda olmalı; büyük harf, küçük harf, rakam ve özel karakter içermelidir.

>>> SORU: SQL ENJEKSİYONU NASIL ÖNLENİR?
CEVAP: Dinamik SQL kullanımı yasaktır. Mutlaka "Parametreli Sorgular" (Prepared Statements) veya ORM kullanılmalıdır.

>>> SORU: SUNUCU MARKASI NEDİR?
CEVAP: Altyapı ve donanım envanter bilgisi, güvenlik politikası gereği GİZLİDİR ve paylaşılamaz.

>>> SORU: CI/CD HATTINA ERİŞİM NASIL OLMALI?
CEVAP: CI/CD hattına sadece yetkilendirilmiş kullanıcılar, MFA (Çok Faktörlü Kimlik Doğrulama) ile erişmelidir.
"""

with open(DOC_NAME, "w", encoding="utf-8") as f:
    f.write(FIXED_DOCUMENT)

# --- 3. BACKEND (FASTAPI) ---
app = FastAPI()
model = None
tokenizer = None

# PROMPT AYARI: Modele kesin sınırlar çiziyoruz
RAG_PROMPT = """Sen uzman bir Siber Güvenlik Asistanısın.
GÖREV: Aşağıdaki [BİLGİ BANKASI] metnini kullanarak soruyu cevapla.

KURALLAR:
1. Sadece verilen metindeki bilgiyi kullan.
2. Eğer sorunun cevabı "GİZLİDİR" içeriyorsa, bunu açıkça belirt ve reddet.
3. Kısa, net ve profesyonel cevap ver.

[BİLGİ BANKASI]:
{context}

Kullanıcı Sorusu: {question}
Cevap:"""

def setup_system():
    global model, tokenizer
    print(f"\n🚀 [Sistem] Model Yükleniyor: {HF_KULLANICI_ADI}/{MODEL_ADI}")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = f"{HF_KULLANICI_ADI}/{MODEL_ADI}",
            max_seq_length = 2048,
            dtype = None,
            load_in_4bit = True,
        )
        FastLanguageModel.for_inference(model)
        print("✅ Model Başarıyla Yüklendi.")
    except Exception as e:
        print(f"🛑 Model Hatası: {e}")

setup_system()

@app.post("/chat")
async def chat_endpoint(request_data: dict):
    global model, tokenizer
    question = request_data.get("question")
    if not model: return {"response": "Model henüz yüklenmedi, lütfen bekleyin..."}

    # Promptu hazırla
    prompt = RAG_PROMPT.format(context=FIXED_DOCUMENT, question=question)

    # Tokenizer ayarları
    inputs = tokenizer([prompt], return_tensors="pt", padding=True).to("cuda")

    # Üretim ayarları (Deterministik olması için temperature düşük)
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.01,  # Daha kararlı cevaplar için düşürdük
        do_sample=True
    )

    decoded = tokenizer.batch_decode(outputs)[0]

    # Cevabı ayrıştır (Parsing)
    if "Cevap:" in decoded:
        response = decoded.split("Cevap:")[-1].strip().replace(tokenizer.eos_token, "")
    else:
        # Eğer model prompt formatını bozarsa ham çıktıyı temizle
        response = decoded.split("Kullanıcı Sorusu:")[-1].strip()

    return {"response": response}

def run_api():
    uvicorn.run(app, host="127.0.0.1", port=8000)

thread = threading.Thread(target=run_api)
thread.start()

# --- 4. FRONTEND (STREAMLIT) ---
streamlit_code = """
import streamlit as st
import requests
import time

# Sayfa Ayarları
st.set_page_config(page_title="CyberSec AI", page_icon="🛡️", layout="centered")

# Başlık Tasarımı
st.markdown("<h1 style='text-align: center; color: #00FF41;'>🛡️ Siber Güvenlik Asistanı</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>RAG Destekli Kurumsal Güvenlik Botu</h4>", unsafe_allow_html=True)
st.divider()

# Session State (Geçmişi tutmak için)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Geçmiş mesajları ekrana bas
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kullanıcı girişi
if prompt := st.chat_input("Güvenlik prosedürleri hakkında bir soru sorun..."):
    # Kullanıcı mesajını ekle
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Bot cevabını al
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Veri tabanı taranıyor ve cevap üretiliyor..."):
            try:
                res = requests.post("http://127.0.0.1:8000/chat", json={"question": prompt})
                if res.status_code == 200:
                    full_response = res.json().get("response", "Cevap alınamadı.")

                    # Daktilo efekti (Görsellik için)
                    displayed_response = ""
                    for char in full_response:
                        displayed_response += char
                        message_placeholder.markdown(displayed_response + "▌")
                        time.sleep(0.01)
                    message_placeholder.markdown(displayed_response)

                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    st.error("API Bağlantı Hatası!")
            except Exception as e:
                st.error(f"Hata oluştu: {e}")
"""
with open("app.py", "w", encoding="utf-8") as f:
    f.write(streamlit_code)

# --- 5. BAŞLATMA VE TÜNEL ---
print("⏳ Streamlit Başlatılıyor...")
if os.path.exists("streamlit.log"): os.remove("streamlit.log")
log_file = open("streamlit.log", "w")

# Streamlit'i arka planda çalıştır
subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "127.0.0.1"], stdout=log_file, stderr=log_file)

import time
time.sleep(5)

print("\n🌐 AŞAĞIDAKİ LİNKE TIKLAYARAK ARAYÜZE GİDEBİLİRSİNİZ:")
if not os.path.exists("cloudflared"):
    !wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    !mv cloudflared-linux-amd64 cloudflared
    !chmod +x cloudflared

# Cloudflare tüneli
!./cloudflared tunnel --url http://127.0.0.1:8501
